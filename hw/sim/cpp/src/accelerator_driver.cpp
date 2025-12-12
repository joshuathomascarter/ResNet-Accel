/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                       ACCELERATOR_DRIVER.CPP                              ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  Implementation of the ResNet-18 sparse accelerator driver.              ║
 * ║  Replaces: sw/host/accel.py                                              ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include "accelerator_driver.hpp"
#include "axi_master.hpp"
#include "csr_map.hpp"
#include "memory_manager.hpp"
#include "bsr_packer.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstring>
#include <sstream>

namespace resnet_accel {

// =============================================================================
// Construction / Destruction
// =============================================================================

AcceleratorDriver::AcceleratorDriver(Mode mode, uint64_t base_addr)
    : mode_(mode)
    , base_addr_(base_addr)
    , initialized_(false)
    , current_layer_(0)
{
}

AcceleratorDriver::~AcceleratorDriver() {
    // Ensure clean shutdown
    if (initialized_) {
        try {
            if (is_busy()) {
                abort();
            }
        } catch (...) {
            // Ignore exceptions in destructor
        }
    }
}

AcceleratorDriver::AcceleratorDriver(AcceleratorDriver&&) noexcept = default;
AcceleratorDriver& AcceleratorDriver::operator=(AcceleratorDriver&&) noexcept = default;

// =============================================================================
// Backend Creation
// =============================================================================

std::unique_ptr<AXIBackend> AcceleratorDriver::create_backend() {
    switch (mode_) {
        case Mode::FPGA:
            // Real FPGA: use /dev/mem backend for physical address access
            return std::make_unique<DevMemBackend>(base_addr_, 0x10000);
            
        case Mode::SIMULATION:
            // Verilator: backend will be set externally or use software model
            return std::make_unique<SoftwareModelBackend>(base_addr_, 0x10000);
            
        case Mode::SOFTWARE_MODEL:
        default:
            // Pure software simulation
            return std::make_unique<SoftwareModelBackend>(base_addr_, 0x10000);
    }
}

// =============================================================================
// Initialization
// =============================================================================

void AcceleratorDriver::initialize() {
    if (initialized_) {
        return;
    }
    
    // Create AXI backend based on mode
    auto backend = create_backend();
    axi_ = std::make_unique<AXIMaster>(std::move(backend), base_addr_);
    
    // Create memory manager
    memory_ = std::make_unique<MemoryManager>();
    
    // Initialize memory manager based on mode
    if (mode_ == Mode::FPGA) {
        // FPGA: use physical memory regions allocated by kernel driver
        // memory_->initialize_fpga();
    } else {
        // Simulation: use software buffers
        // memory_->initialize_simulation();
    }
    
    // Reset accelerator to known state
    reset();
    
    // Verify communication by reading status register
    uint32_t status = read_reg(csr::STATUS);
    (void)status;  // Suppress unused warning in release builds
    
    initialized_ = true;
    
    std::cout << "[AcceleratorDriver] Initialized in " 
              << (mode_ == Mode::FPGA ? "FPGA" : 
                  mode_ == Mode::SIMULATION ? "SIMULATION" : "SOFTWARE_MODEL")
              << " mode at base address 0x" << std::hex << base_addr_ << std::dec 
              << std::endl;
}

void AcceleratorDriver::reset() {
    // Write abort to stop any pending operation (W1P - write 1 pulse)
    write_reg(csr::CTRL, csr::CTRL_ABORT);
    
    // Small delay for reset to propagate through synchronizers
    std::this_thread::sleep_for(std::chrono::microseconds(10));
    
    // Clear control register
    write_reg(csr::CTRL, 0);
    
    // Clear any sticky error flags (W1C - write 1 to clear)
    clear_errors();
    
    // Reset dimension registers to zero
    write_reg(csr::DIMS_M, 0);
    write_reg(csr::DIMS_N, 0);
    write_reg(csr::DIMS_K, 0);
    
    // Reset tile dimensions to systolic array size (safe defaults)
    write_reg(csr::TILES_Tm, csr::SYSTOLIC_ROWS);
    write_reg(csr::TILES_Tn, csr::SYSTOLIC_COLS);
    write_reg(csr::TILES_Tk, csr::BLOCK_SIZE);
    
    // Reset quantization scales to 1.0f (identity scaling)
    write_reg(csr::SCALE_Sa, float_to_bits(1.0f));
    write_reg(csr::SCALE_Sw, float_to_bits(1.0f));
    
    // Wait for busy to clear
    for (int i = 0; i < 100; ++i) {
        if (!is_busy()) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    
    std::cerr << "[AcceleratorDriver] Warning: Accelerator still busy after reset" << std::endl;
}

uint32_t AcceleratorDriver::get_version() {
    // Version register would be at a fixed offset
    // For now, return a placeholder indicating v1.0.0
    return 0x00010000;
}

// =============================================================================
// Layer Configuration
// =============================================================================

void AcceleratorDriver::configure_layer(const LayerConfig& config) {
    // Validate configuration before writing to hardware
    validate_config(config);
    
    // =========================================================================
    // Step 1: Write matrix dimensions (M x N x K matmul or im2col geometry)
    // =========================================================================
    set_dimensions(config.M, config.N, config.K);
    
    // =========================================================================
    // Step 2: Write tile dimensions for tiled execution
    // =========================================================================
    set_tile_dimensions(config.Tm, config.Tn, config.Tk);
    
    // =========================================================================
    // Step 3: Write quantization scales (FP32 → INT8 dequantization)
    // =========================================================================
    set_scales(config.scale_activation, config.scale_weight);
    
    // =========================================================================
    // Step 4: Configure DMA buffer addresses for data transfer
    // =========================================================================
    if (config.act_addr != 0) {
        set_activation_buffer(config.act_addr, config.activation_size_bytes());
    }
    
    if (config.wgt_addr != 0) {
        set_weight_buffer(config.wgt_addr, config.weight_size_bytes_dense());
    }
    
    if (config.out_addr != 0) {
        set_output_buffer(config.out_addr);
    }
    
    // =========================================================================
    // Step 5: Configure BSR sparse format parameters (if sparse mode)
    // =========================================================================
    if (config.is_sparse && config.bsr_ptr_addr != 0) {
        set_bsr_buffers(config.bsr_ptr_addr, config.bsr_idx_addr, config.bsr_nnz_blocks);
    }
}

void AcceleratorDriver::set_dimensions(uint32_t M, uint32_t N, uint32_t K) {
    write_reg(csr::DIMS_M, M);
    write_reg(csr::DIMS_N, N);
    write_reg(csr::DIMS_K, K);
}

void AcceleratorDriver::set_tile_dimensions(uint32_t Tm, uint32_t Tn, uint32_t Tk) {
    write_reg(csr::TILES_Tm, Tm);
    write_reg(csr::TILES_Tn, Tn);
    write_reg(csr::TILES_Tk, Tk);
}

void AcceleratorDriver::set_scales(float scale_act, float scale_wgt) {
    write_reg(csr::SCALE_Sa, float_to_bits(scale_act));
    write_reg(csr::SCALE_Sw, float_to_bits(scale_wgt));
}

void AcceleratorDriver::validate_config(const LayerConfig& config) {
    // Check for zero dimensions
    if (config.M == 0 || config.N == 0 || config.K == 0) {
        throw AcceleratorError(
            AcceleratorError::Code::INVALID_CONFIG,
            "Matrix dimensions (M, N, K) must be non-zero");
    }
    
    // Check for zero tile dimensions
    if (config.Tm == 0 || config.Tn == 0 || config.Tk == 0) {
        throw AcceleratorError(
            AcceleratorError::Code::INVALID_CONFIG,
            "Tile dimensions (Tm, Tn, Tk) must be non-zero");
    }
    
    // Check tile dimensions fit within systolic array
    if (config.Tm > csr::SYSTOLIC_ROWS || config.Tn > csr::SYSTOLIC_COLS) {
        throw AcceleratorError(
            AcceleratorError::Code::INVALID_CONFIG,
            "Tile dimensions exceed systolic array size (16x16)");
    }
    
    // Check for invalid quantization scales
    if (config.scale_activation <= 0.0f || config.scale_weight <= 0.0f) {
        throw AcceleratorError(
            AcceleratorError::Code::INVALID_CONFIG,
            "Quantization scales must be positive");
    }
    
    // Warn if dimensions are not tile-aligned (not an error, just suboptimal)
    if (config.M % config.Tm != 0 || config.N % config.Tn != 0 || config.K % config.Tk != 0) {
        std::cerr << "[AcceleratorDriver] Warning: Dimensions not tile-aligned, "
                  << "performance may be suboptimal" << std::endl;
    }
}

// =============================================================================
// DMA Buffer Configuration
// =============================================================================

void AcceleratorDriver::set_activation_buffer(uint64_t phys_addr, uint32_t size_bytes) {
    // Set source address for activation DMA
    write_reg(csr::ACT_DMA_SRC_ADDR, static_cast<uint32_t>(phys_addr));
    // Set transfer length
    write_reg(csr::ACT_DMA_LEN, size_bytes);
}

void AcceleratorDriver::set_weight_buffer(uint64_t phys_addr, uint32_t size_bytes) {
    // Set source address for weight DMA (also used for BSR data)
    write_reg(csr::DMA_SRC_ADDR, static_cast<uint32_t>(phys_addr));
    // Set transfer length
    write_reg(csr::DMA_XFER_LEN, size_bytes);
}

void AcceleratorDriver::set_output_buffer(uint64_t phys_addr) {
    // Set destination address for output DMA
    write_reg(csr::DMA_DST_ADDR, static_cast<uint32_t>(phys_addr));
}

void AcceleratorDriver::set_bsr_buffers(uint64_t ptr_addr, uint64_t idx_addr, uint32_t nnz_blocks) {
    // BSR format uses separate addresses for row_ptr and col_idx arrays
    // These could be packed into the DMA address registers or use dedicated BSR registers
    // For now, store in DMA registers (actual RTL may need dedicated registers)
    
    write_reg(csr::DMA_SRC_ADDR, static_cast<uint32_t>(ptr_addr));
    
    // Store nnz_blocks count - this determines number of MAC operations
    // Could be stored in a dedicated register or computed from dimensions
    (void)idx_addr;
    (void)nnz_blocks;
    
    // TODO: Add BSR-specific registers to CSR map when RTL is extended
}

// =============================================================================
// Execution Control
// =============================================================================

void AcceleratorDriver::start() {
    // Clear any previous done flags (W1C - write 1 to clear)
    write_reg(csr::STATUS, csr::STATUS_DONE_TILE);
    
    // Write start pulse (W1P - write 1 to pulse, auto-clears)
    write_reg(csr::CTRL, csr::CTRL_START);
}

void AcceleratorDriver::abort() {
    // Write abort pulse to stop current operation
    write_reg(csr::CTRL, csr::CTRL_ABORT);
    
    // Wait for busy to clear (hardware should abort within a few cycles)
    for (int i = 0; i < 100; ++i) {
        if (!is_busy()) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    
    std::cerr << "[AcceleratorDriver] Warning: Accelerator still busy after abort" << std::endl;
}

void AcceleratorDriver::wait_done(uint32_t timeout_ms) {
    auto start_time = std::chrono::steady_clock::now();
    auto timeout = std::chrono::milliseconds(timeout_ms);
    
    while (true) {
        uint32_t status = read_status();
        
        // =====================================================================
        // Check for successful completion
        // =====================================================================
        if (status & csr::STATUS_DONE_TILE) {
            return;
        }
        
        // =====================================================================
        // Check for error condition
        // =====================================================================
        if (status & csr::STATUS_ERR_ILLEGAL) {
            throw AcceleratorError(
                AcceleratorError::Code::ILLEGAL_COMMAND,
                "Accelerator reported illegal command error (bad register access or invalid config)");
        }
        
        // =====================================================================
        // Check for timeout
        // =====================================================================
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed > timeout) {
            throw AcceleratorError(
                AcceleratorError::Code::TIMEOUT,
                "Timeout waiting for accelerator completion after " + 
                std::to_string(timeout_ms) + "ms");
        }
        
        // =====================================================================
        // Poll delay to reduce bus traffic (adjust based on expected latency)
        // =====================================================================
        if (mode_ == Mode::FPGA) {
            // FPGA: longer delay to reduce AXI bus traffic
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        } else {
            // Simulation: shorter delay for faster testing
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}

void AcceleratorDriver::run_layer(const LayerConfig& config, 
                                   const int8_t* input, 
                                   int32_t* output) {
    // =========================================================================
    // Step 1: Configure layer parameters in CSR
    // =========================================================================
    configure_layer(config);
    
    // =========================================================================
    // Step 2: Copy input data to DMA buffer (if using MemoryManager)
    // =========================================================================
    if (memory_ && input) {
        // memory_->load_activations(input, config.activation_size_bytes());
    }
    
    // =========================================================================
    // Step 3: Start DMA transfers for activations and weights
    // =========================================================================
    if (config.act_addr != 0) {
        start_activation_dma(config.act_addr, config.activation_size_bytes());
    }
    
    if (config.wgt_addr != 0) {
        start_weight_dma(config.wgt_addr, config.weight_size_bytes_dense());
    }
    
    // =========================================================================
    // Step 4: Wait for DMA transfers to complete
    // =========================================================================
    wait_dma_done();
    
    // =========================================================================
    // Step 5: Start systolic array computation
    // =========================================================================
    start();
    
    // =========================================================================
    // Step 6: Wait for computation to complete
    // =========================================================================
    wait_done();
    
    // =========================================================================
    // Step 7: Copy output data from DMA buffer (if using MemoryManager)
    // =========================================================================
    if (memory_ && output) {
        // memory_->read_outputs(output, config.output_size_bytes());
    }
}

void AcceleratorDriver::run_layer(size_t layer_idx, 
                                   const int8_t* input, 
                                   int32_t* output) {
    if (layer_idx >= layer_configs_.size()) {
        throw AcceleratorError(
            AcceleratorError::Code::INVALID_CONFIG,
            "Layer index " + std::to_string(layer_idx) + 
            " out of range (have " + std::to_string(layer_configs_.size()) + " layers)");
    }
    
    current_layer_ = layer_idx;
    run_layer(layer_configs_[layer_idx], input, output);
}

// =============================================================================
// Status Checking
// =============================================================================

bool AcceleratorDriver::is_busy() {
    uint32_t status = read_status();
    return (status & csr::STATUS_BUSY) != 0;
}

bool AcceleratorDriver::is_done() {
    uint32_t status = read_status();
    return (status & csr::STATUS_DONE_TILE) != 0;
}

bool AcceleratorDriver::has_error() {
    uint32_t status = read_status();
    return (status & csr::STATUS_ERR_ILLEGAL) != 0;
}

uint32_t AcceleratorDriver::read_status() {
    return read_reg(csr::STATUS);
}

void AcceleratorDriver::clear_errors() {
    // Write 1 to clear (W1C) the sticky error and done bits
    write_reg(csr::STATUS, csr::STATUS_DONE_TILE | csr::STATUS_ERR_ILLEGAL);
}

void AcceleratorDriver::dump_status() {
    uint32_t status = read_status();
    uint32_t ctrl = read_reg(csr::CTRL);
    uint32_t m = read_reg(csr::DIMS_M);
    uint32_t n = read_reg(csr::DIMS_N);
    uint32_t k = read_reg(csr::DIMS_K);
    uint32_t tm = read_reg(csr::TILES_Tm);
    uint32_t tn = read_reg(csr::TILES_Tn);
    uint32_t tk = read_reg(csr::TILES_Tk);
    
    std::cout << "╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    ACCELERATOR STATUS DUMP                     ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
    
    std::cout << "║ Status: 0x" << std::hex << std::setfill('0') << std::setw(8) << status 
              << std::dec << std::setfill(' ') << std::setw(42) << "║" << std::endl;
    std::cout << "║   Busy:     " << std::setw(4) << ((status & csr::STATUS_BUSY) ? "YES" : "no")
              << std::setw(49) << "║" << std::endl;
    std::cout << "║   Done:     " << std::setw(4) << ((status & csr::STATUS_DONE_TILE) ? "YES" : "no")
              << std::setw(49) << "║" << std::endl;
    std::cout << "║   Error:    " << std::setw(4) << ((status & csr::STATUS_ERR_ILLEGAL) ? "YES" : "no")
              << std::setw(49) << "║" << std::endl;
    
    std::cout << "╠────────────────────────────────────────────────────────────────╣" << std::endl;
    
    std::cout << "║ Control: 0x" << std::hex << std::setfill('0') << std::setw(8) << ctrl 
              << std::dec << std::setfill(' ') << std::setw(41) << "║" << std::endl;
    std::cout << "║   IRQ Enable: " << std::setw(4) << ((ctrl & csr::CTRL_IRQ_EN) ? "YES" : "no")
              << std::setw(47) << "║" << std::endl;
    
    std::cout << "╠────────────────────────────────────────────────────────────────╣" << std::endl;
    
    std::cout << "║ Dimensions: M=" << std::setw(5) << m 
              << "  N=" << std::setw(5) << n 
              << "  K=" << std::setw(5) << k
              << std::setw(33) << "║" << std::endl;
    std::cout << "║ Tiles:      Tm=" << std::setw(4) << tm 
              << "  Tn=" << std::setw(4) << tn 
              << "  Tk=" << std::setw(4) << tk
              << std::setw(33) << "║" << std::endl;
    
    std::cout << "╠────────────────────────────────────────────────────────────────╣" << std::endl;
    
    // Read and display performance counters
    PerfCounters perf = read_perf_counters();
    std::cout << "║ Performance Counters:" << std::setw(44) << "║" << std::endl;
    std::cout << "║   Total Cycles:  " << std::setw(12) << perf.total_cycles 
              << std::setw(36) << "║" << std::endl;
    std::cout << "║   Active Cycles: " << std::setw(12) << perf.active_cycles 
              << std::setw(36) << "║" << std::endl;
    std::cout << "║   Idle Cycles:   " << std::setw(12) << perf.idle_cycles 
              << std::setw(36) << "║" << std::endl;
    std::cout << "║   Utilization:   " << std::setw(10) << std::fixed << std::setprecision(1) 
              << (perf.utilization() * 100.0f) << "%" << std::setw(36) << "║" << std::endl;
    std::cout << "║   Cache Hits:    " << std::setw(12) << perf.cache_hits 
              << std::setw(36) << "║" << std::endl;
    std::cout << "║   Cache Misses:  " << std::setw(12) << perf.cache_misses 
              << std::setw(36) << "║" << std::endl;
    std::cout << "║   Hit Rate:      " << std::setw(10) << std::fixed << std::setprecision(1)
              << (perf.cache_hit_rate() * 100.0f) << "%" << std::setw(36) << "║" << std::endl;
    
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;
}

// =============================================================================
// Performance Monitoring
// =============================================================================

PerfCounters AcceleratorDriver::read_perf_counters() {
    PerfCounters counters;
    
    counters.total_cycles  = read_reg(csr::PERF_TOTAL);
    counters.active_cycles = read_reg(csr::PERF_ACTIVE);
    counters.idle_cycles   = read_reg(csr::PERF_IDLE);
    counters.cache_hits    = read_reg(csr::PERF_CACHE_HITS);
    counters.cache_misses  = read_reg(csr::PERF_CACHE_MISSES);
    counters.decode_count  = read_reg(csr::PERF_DECODE_COUNT);
    
    return counters;
}

void AcceleratorDriver::reset_perf_counters() {
    // Performance counters are automatically reset on start in most designs
    // If manual reset is needed, add a PERF_CTRL register with reset bit
}

// =============================================================================
// Weight Management
// =============================================================================

void AcceleratorDriver::load_weights_bsr(const std::string& weight_dir) {
    // Load all layer weights from directory
    // Expected files: layer_0.bsr, layer_1.bsr, etc.
    
    layer_weights_.clear();
    layer_configs_.clear();
    
    for (size_t i = 0; ; ++i) {
        std::ostringstream filename;
        filename << weight_dir << "/layer_" << i << ".bsr";
        
        std::ifstream file(filename.str(), std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            break;  // No more layers
        }
        
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<uint8_t> data(size);
        if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
            std::cerr << "[AcceleratorDriver] Error reading " << filename.str() << std::endl;
            break;
        }
        
        layer_weights_.push_back(std::move(data));
        
        std::cout << "[AcceleratorDriver] Loaded layer " << i 
                  << " weights (" << size << " bytes)" << std::endl;
    }
    
    std::cout << "[AcceleratorDriver] Loaded " << layer_weights_.size() 
              << " layers from " << weight_dir << std::endl;
}

void AcceleratorDriver::set_layer_weights(size_t layer_idx, const BSRMatrix& bsr) {
    (void)layer_idx;
    (void)bsr;
    // TODO: Pack BSR matrix using bsr_packer_ and store in layer_weights_
}

void AcceleratorDriver::set_layer_weights(size_t layer_idx, const void* bsr_data, size_t size) {
    if (layer_idx >= layer_weights_.size()) {
        layer_weights_.resize(layer_idx + 1);
    }
    
    layer_weights_[layer_idx].resize(size);
    std::memcpy(layer_weights_[layer_idx].data(), bsr_data, size);
}

// =============================================================================
// DMA Control
// =============================================================================

void AcceleratorDriver::start_weight_dma(uint64_t src_addr, uint32_t len) {
    // Configure weight DMA source address and length
    write_reg(csr::DMA_SRC_ADDR, static_cast<uint32_t>(src_addr));
    write_reg(csr::DMA_XFER_LEN, len);
    
    // Start DMA transfer (W1P - write 1 pulse)
    write_reg(csr::DMA_CTRL, csr::DMA_CTRL_START);
}

void AcceleratorDriver::start_activation_dma(uint64_t src_addr, uint32_t len) {
    // Configure activation DMA source address and length
    write_reg(csr::ACT_DMA_SRC_ADDR, static_cast<uint32_t>(src_addr));
    write_reg(csr::ACT_DMA_LEN, len);
    
    // Start DMA transfer (W1P - write 1 pulse)
    write_reg(csr::ACT_DMA_CTRL, csr::DMA_CTRL_START);
}

void AcceleratorDriver::wait_dma_done(uint32_t timeout_ms) {
    auto start_time = std::chrono::steady_clock::now();
    auto timeout = std::chrono::milliseconds(timeout_ms);
    
    while (true) {
        if (!is_dma_busy()) {
            return;
        }
        
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (elapsed > timeout) {
            throw AcceleratorError(
                AcceleratorError::Code::TIMEOUT,
                "Timeout waiting for DMA completion after " + 
                std::to_string(timeout_ms) + "ms");
        }
        
        // Small delay between polls
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

bool AcceleratorDriver::is_dma_busy() {
    uint32_t dma_ctrl = read_reg(csr::DMA_CTRL);
    uint32_t act_dma_ctrl = read_reg(csr::ACT_DMA_CTRL);
    
    // Check busy bits for both weight and activation DMA engines
    return ((dma_ctrl & csr::DMA_CTRL_BUSY) != 0) ||
           ((act_dma_ctrl & csr::DMA_CTRL_BUSY) != 0);
}

// =============================================================================
// Direct Register Access (Low-Level)
// =============================================================================

void AcceleratorDriver::write_reg(uint32_t offset, uint32_t value) {
    if (axi_) {
        axi_->write_reg(offset, value);
    }
}

uint32_t AcceleratorDriver::read_reg(uint32_t offset) {
    if (axi_) {
        return axi_->read_reg(offset);
    }
    return 0;
}

} // namespace resnet_accel
