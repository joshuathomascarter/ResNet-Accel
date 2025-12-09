/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                     ACCELERATOR_DRIVER.CPP                                ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  IMPLEMENTS: accelerator_driver.hpp                                       ║
 * ║  REPLACES: sw/host/accel.py                                               ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  WHAT YOU NEED TO IMPLEMENT:                                              ║
 * ║                                                                           ║
 * ║  1. Constructor: AcceleratorDriver(Mode mode)                             ║
 * ║     - Create appropriate AXI backend based on mode                        ║
 * ║     - FPGA: DevMemBackend with physical address 0x43C00000                ║
 * ║     - SIMULATION: VerilatorBackend or SoftwareModelBackend                ║
 * ║     - Create MemoryManager and BSRPacker instances                        ║
 * ║                                                                           ║
 * ║  2. initialize()                                                          ║
 * ║     - Call reset()                                                        ║
 * ║     - Read and verify VERSION register                                    ║
 * ║     - Allocate maximum-size DMA buffers                                   ║
 * ║     - Return error if version mismatch                                    ║
 * ║                                                                           ║
 * ║  3. reset()                                                               ║
 * ║     - Write CTRL_RESET bit to CTRL register                               ║
 * ║     - Wait a few cycles (barrier)                                         ║
 * ║     - Clear CTRL register                                                 ║
 * ║     - Wait for STATUS.BUSY to clear                                       ║
 * ║                                                                           ║
 * ║  4. configure_layer(LayerConfig)                                          ║
 * ║     - Write all layer parameters to registers:                            ║
 * ║       IN_CHANNELS, OUT_CHANNELS, IN_HEIGHT, IN_WIDTH                      ║
 * ║       KERNEL_SIZE, STRIDE, PADDING                                        ║
 * ║       ACT_SCALE, WGT_SCALE, OUT_SCALE                                     ║
 * ║                                                                           ║
 * ║  5. run_layer(layer_idx, input, output)                                   ║
 * ║     - Get layer config and weights                                        ║
 * ║     - Load input to activation buffer via memory manager                  ║
 * ║     - Configure layer parameters                                          ║
 * ║     - Configure BSR parameters (nnz_blocks, block_rows, addresses)        ║
 * ║     - Write buffer addresses to registers                                 ║
 * ║     - Write CTRL_START | CTRL_SPARSE_MODE                                 ║
 * ║     - Call wait_done()                                                    ║
 * ║     - Read output from output buffer                                      ║
 * ║                                                                           ║
 * ║  6. wait_done(timeout_ms)                                                 ║
 * ║     - Poll STATUS register until DONE bit set                             ║
 * ║     - Check ERROR bit and throw if set                                    ║
 * ║     - Return false if timeout exceeded                                    ║
 * ║     - For simulation: just loop                                           ║
 * ║     - For FPGA: use sleep between polls to reduce bus traffic             ║
 * ║                                                                           ║
 * ║  7. read_perf_counters()                                                  ║
 * ║     - Read CYCLE_COUNT, COMPUTE_CYC, STALL_CYC, MAC_OPS                   ║
 * ║     - Return PerfCounters struct                                          ║
 * ║                                                                           ║
 * ║  REGISTER OFFSETS (from hw/rtl/csr.sv):                                   ║
 * ║     CTRL = 0x00, STATUS = 0x04                                            ║
 * ║     ACT_BASE = 0x10, WGT_BASE = 0x14, OUT_BASE = 0x18                     ║
 * ║     IN_CHANNELS = 0x24, OUT_CHANNELS = 0x28                               ║
 * ║     IN_HEIGHT = 0x2C, IN_WIDTH = 0x30                                     ║
 * ║     KERNEL_SIZE = 0x34, STRIDE = 0x38, PADDING = 0x3C                     ║
 * ║     BSR_NNZ = 0x70, BSR_ROWS = 0x74, BSR_PTR = 0x78, BSR_IDX = 0x7C       ║
 * ║     CYCLE_COUNT = 0x50, COMPUTE_CYC = 0x54, STALL_CYC = 0x58, MAC_OPS=0x5C║
 * ║                                                                           ║
 * ║  CTRL BITS:                                                               ║
 * ║     START = bit 0, RESET = bit 1, SPARSE_MODE = bit 2                     ║
 * ║                                                                           ║
 * ║  STATUS BITS:                                                             ║
 * ║     BUSY = bit 0, DONE = bit 1, ERROR = bit 2                             ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include "accelerator_driver.hpp"
#include "axi_master.hpp"
#include "memory_manager.hpp"
#include "bsr_packer.hpp"

#include <stdexcept>
#include <thread>
#include <chrono>

// Register offsets
namespace reg {
    constexpr uint32_t CTRL = 0x00;
    constexpr uint32_t STATUS = 0x04;
    constexpr uint32_t ACT_BASE = 0x10;
    constexpr uint32_t WGT_BASE = 0x14;
    constexpr uint32_t OUT_BASE = 0x18;
    constexpr uint32_t IN_CHANNELS = 0x24;
    constexpr uint32_t OUT_CHANNELS = 0x28;
    constexpr uint32_t IN_HEIGHT = 0x2C;
    constexpr uint32_t IN_WIDTH = 0x30;
    constexpr uint32_t KERNEL_SIZE = 0x34;
    constexpr uint32_t STRIDE = 0x38;
    constexpr uint32_t PADDING = 0x3C;
    constexpr uint32_t CYCLE_COUNT = 0x50;
    constexpr uint32_t COMPUTE_CYC = 0x54;
    constexpr uint32_t STALL_CYC = 0x58;
    constexpr uint32_t MAC_OPS = 0x5C;
    constexpr uint32_t VERSION = 0x68;
    constexpr uint32_t BSR_NNZ = 0x70;
    constexpr uint32_t BSR_ROWS = 0x74;
    constexpr uint32_t BSR_PTR = 0x78;
    constexpr uint32_t BSR_IDX = 0x7C;
}

// Control bits
namespace ctrl {
    constexpr uint32_t START = (1 << 0);
    constexpr uint32_t RESET = (1 << 1);
    constexpr uint32_t SPARSE_MODE = (1 << 2);
}

// Status bits
namespace status {
    constexpr uint32_t BUSY = (1 << 0);
    constexpr uint32_t DONE = (1 << 1);
    constexpr uint32_t ERROR = (1 << 2);
}

// FPGA base address
constexpr uint64_t ACCEL_BASE_ADDR = 0x43C00000;
constexpr size_t ACCEL_REG_SIZE = 0x10000;

// -----------------------------------------------------------------------------
// Constructor / Destructor
// -----------------------------------------------------------------------------

AcceleratorDriver::AcceleratorDriver(Mode mode) : mode_(mode) {
    // TODO: Create appropriate backend based on mode
    //
    // if (mode == Mode::FPGA) {
    //     auto backend = std::make_unique<DevMemBackend>(ACCEL_BASE_ADDR, ACCEL_REG_SIZE);
    //     axi_ = std::make_unique<AXIMaster>(std::move(backend), ACCEL_BASE_ADDR);
    // } else if (mode == Mode::SIMULATION) {
    //     auto backend = std::make_unique<SoftwareModelBackend>(0, ACCEL_REG_SIZE);
    //     axi_ = std::make_unique<AXIMaster>(std::move(backend), 0);
    // }
    //
    // memory_ = std::make_unique<MemoryManager>(
    //     mode == Mode::FPGA ? MemoryManager::Mode::FPGA : MemoryManager::Mode::SIMULATION
    // );
    //
    // bsr_packer_ = std::make_unique<BSRPacker>();
}

AcceleratorDriver::~AcceleratorDriver() = default;

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

void AcceleratorDriver::initialize() {
    // TODO: Implement
    // 1. reset();
    // 2. uint32_t version = get_version();
    // 3. Verify version matches expected
    // 4. memory_->allocate_max_buffers();
}

void AcceleratorDriver::reset() {
    // TODO: Implement
    // write_reg(reg::CTRL, ctrl::RESET);
    // axi_->get_backend()->barrier();
    // write_reg(reg::CTRL, 0);
    // 
    // // Wait for not busy
    // while (read_reg(reg::STATUS) & status::BUSY) {
    //     // spin
    // }
}

uint32_t AcceleratorDriver::get_version() {
    // TODO: Implement
    // return read_reg(reg::VERSION);
    return 0;
}

// -----------------------------------------------------------------------------
// Layer Configuration
// -----------------------------------------------------------------------------

void AcceleratorDriver::configure_layer(const LayerConfig& config) {
    // TODO: Implement
    // write_reg(reg::IN_CHANNELS, config.in_channels);
    // write_reg(reg::OUT_CHANNELS, config.out_channels);
    // write_reg(reg::IN_HEIGHT, config.in_height);
    // write_reg(reg::IN_WIDTH, config.in_width);
    // write_reg(reg::KERNEL_SIZE, config.kernel_size);
    // write_reg(reg::STRIDE, config.stride);
    // write_reg(reg::PADDING, config.padding);
}

// -----------------------------------------------------------------------------
// Layer Execution
// -----------------------------------------------------------------------------

void AcceleratorDriver::run_layer(size_t layer_idx, 
                                   const int8_t* input, 
                                   int32_t* output) {
    // TODO: Implement full layer execution
    //
    // 1. Get layer configuration for layer_idx
    // 2. Calculate sizes
    //    size_t in_size = in_channels * in_height * in_width;
    //    size_t out_size = out_channels * out_height * out_width;
    //
    // 3. Load input to DMA buffer
    //    memory_->load_activations(input, in_size);
    //
    // 4. Configure layer
    //    configure_layer(layer_config);
    //
    // 5. Configure BSR parameters
    //    write_reg(reg::BSR_NNZ, bsr.nnz_blocks);
    //    write_reg(reg::BSR_ROWS, bsr.num_block_rows);
    //    write_reg(reg::BSR_PTR, memory_->get_bsr_phys_addr());
    //    write_reg(reg::BSR_IDX, memory_->get_bsr_phys_addr() + ptr_offset);
    //
    // 6. Set buffer addresses
    //    write_reg(reg::ACT_BASE, memory_->get_act_phys_addr());
    //    write_reg(reg::WGT_BASE, memory_->get_wgt_phys_addr());
    //    write_reg(reg::OUT_BASE, memory_->get_out_phys_addr());
    //
    // 7. Start computation
    //    write_reg(reg::CTRL, ctrl::START | ctrl::SPARSE_MODE);
    //
    // 8. Wait for completion
    //    if (!wait_done(5000)) {
    //        throw std::runtime_error("Timeout waiting for accelerator");
    //    }
    //
    // 9. Read output
    //    memory_->read_outputs(output, out_size * sizeof(int32_t));
}

void AcceleratorDriver::wait_done(uint32_t timeout_ms) {
    // TODO: Implement
    //
    // auto start = std::chrono::steady_clock::now();
    //
    // while (true) {
    //     uint32_t stat = read_reg(reg::STATUS);
    //     
    //     if (stat & status::ERROR) {
    //         throw std::runtime_error("Accelerator error");
    //     }
    //     
    //     if (stat & status::DONE) {
    //         return;  // Success
    //     }
    //     
    //     auto now = std::chrono::steady_clock::now();
    //     auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
    //     if (elapsed.count() > timeout_ms) {
    //         throw std::runtime_error("Timeout");
    //     }
    //     
    //     // Sleep briefly for FPGA to reduce bus traffic
    //     if (mode_ == Mode::FPGA) {
    //         std::this_thread::sleep_for(std::chrono::microseconds(100));
    //     }
    // }
}

// -----------------------------------------------------------------------------
// Weight Management
// -----------------------------------------------------------------------------

void AcceleratorDriver::load_weights_bsr(const std::string& weight_dir) {
    // TODO: Implement
    // For each layer:
    //   1. Load weight .npy file
    //   2. Convert to BSR using bsr_packer_
    //   3. Store for later use
}

void AcceleratorDriver::set_layer_weights(size_t layer_idx, 
                                           const void* bsr_data, 
                                           size_t size) {
    // TODO: Implement
    // memory_->load_weights(bsr_data, size);
}

// -----------------------------------------------------------------------------
// Buffer Management
// -----------------------------------------------------------------------------

void AcceleratorDriver::set_activation_buffer(uint64_t phys_addr) {
    write_reg(reg::ACT_BASE, static_cast<uint32_t>(phys_addr));
}

void AcceleratorDriver::set_weight_buffer(uint64_t phys_addr) {
    write_reg(reg::WGT_BASE, static_cast<uint32_t>(phys_addr));
}

void AcceleratorDriver::set_output_buffer(uint64_t phys_addr) {
    write_reg(reg::OUT_BASE, static_cast<uint32_t>(phys_addr));
}

// -----------------------------------------------------------------------------
// Status and Performance
// -----------------------------------------------------------------------------

bool AcceleratorDriver::is_busy() {
    return (read_reg(reg::STATUS) & status::BUSY) != 0;
}

bool AcceleratorDriver::is_done() {
    return (read_reg(reg::STATUS) & status::DONE) != 0;
}

bool AcceleratorDriver::has_error() {
    return (read_reg(reg::STATUS) & status::ERROR) != 0;
}

PerfCounters AcceleratorDriver::read_perf_counters() {
    PerfCounters pc;
    // TODO: Implement
    // pc.total_cycles = read_reg(reg::CYCLE_COUNT);
    // pc.compute_cycles = read_reg(reg::COMPUTE_CYC);
    // pc.stall_cycles = read_reg(reg::STALL_CYC);
    // pc.mac_operations = read_reg(reg::MAC_OPS);
    return pc;
}

void AcceleratorDriver::dump_status() {
    uint32_t stat = read_reg(reg::STATUS);
    printf("Accelerator Status: 0x%08X\n", stat);
    printf("  BUSY:  %s\n", (stat & status::BUSY) ? "YES" : "no");
    printf("  DONE:  %s\n", (stat & status::DONE) ? "YES" : "no");
    printf("  ERROR: %s\n", (stat & status::ERROR) ? "YES" : "no");
}

// -----------------------------------------------------------------------------
// Private Helpers
// -----------------------------------------------------------------------------

void AcceleratorDriver::write_reg(uint32_t offset, uint32_t value) {
    // TODO: Implement
    // axi_->write_reg(offset, value);
}

uint32_t AcceleratorDriver::read_reg(uint32_t offset) {
    // TODO: Implement
    // return axi_->read_reg(offset);
    return 0;
}
