/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                       ACCELERATOR_DRIVER.HPP                              ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  C++ Host Driver for INT8 Sparse Systolic Array Accelerator              ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  REPLACES: sw/host/accel.py (Python prototype)                           ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  OVERVIEW:                                                                ║
 * ║  ---------                                                                ║
 * ║  This class provides the host-side interface for controlling the FPGA    ║
 * ║  accelerator. It handles:                                                ║
 * ║    - Register configuration via memory-mapped CSRs                       ║
 * ║    - DMA setup and buffer management                                     ║
 * ║    - Execution control (start/stop/wait)                                 ║
 * ║    - Performance monitoring                                              ║
 * ║    - Weight loading (BSR sparse format)                                  ║
 * ║                                                                           ║
 * ║  OPERATION MODES:                                                         ║
 * ║  ----------------                                                         ║
 * ║    FPGA:           Real hardware on Zynq-7020 via /dev/mem mmap          ║
 * ║    SIMULATION:     Verilator model with AXI VPI interface                ║
 * ║    SOFTWARE_MODEL: Pure C++ golden model (no RTL)                        ║
 * ║                                                                           ║
 * ║  TYPICAL USAGE:                                                           ║
 * ║  --------------                                                           ║
 * ║    AcceleratorDriver driver(Mode::FPGA);                                 ║
 * ║    driver.initialize();                                                   ║
 * ║                                                                           ║
 * ║    // Load pre-quantized weights (BSR format)                            ║
 * ║    driver.load_weights_bsr("data/bsr_export/");                          ║
 * ║                                                                           ║
 * ║    // Configure and run a layer                                          ║
 * ║    LayerConfig cfg;                                                       ║
 * ║    cfg.M = 196; cfg.N = 1; cfg.K = 512;  // Example: fc1 layer           ║
 * ║    cfg.is_sparse = true;                                                 ║
 * ║    driver.run_layer(cfg, input_data, output_data);                       ║
 * ║                                                                           ║
 * ║    // Read performance                                                    ║
 * ║    auto perf = driver.read_perf_counters();                              ║
 * ║    printf("Utilization: %.1f%%\n", perf.utilization() * 100);            ║
 * ║                                                                           ║
 * ║  THREAD SAFETY:                                                           ║
 * ║  --------------                                                           ║
 * ║  This class is NOT thread-safe. External synchronization required        ║
 * ║  for multi-threaded access.                                              ║
 * ║                                                                           ║
 * ║  DEPENDENCIES:                                                            ║
 * ║  -------------                                                            ║
 * ║    - csr_map.hpp:        Register address definitions                    ║
 * ║    - memory_manager.hpp: DMA buffer allocation                           ║
 * ║    - bsr_packer.hpp:     Sparse weight format utilities                  ║
 * ║    - axi_master.hpp:     AXI transaction interface                       ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#ifndef ACCELERATOR_DRIVER_HPP
#define ACCELERATOR_DRIVER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>

#include "csr_map.hpp"

namespace resnet_accel {

// Forward declarations for dependency injection
class AXIMaster;      // AXI transaction interface (hw/sim/cpp/include/axi_master.hpp)
class AXIBackend;     // Backend implementation (FPGA mmap, Verilator VPI, etc.)
class MemoryManager;  // DMA buffer allocation (hw/sim/cpp/include/memory_manager.hpp)
class BSRPacker;      // BSR format utilities (hw/sim/cpp/include/bsr_packer.hpp)
struct BSRMatrix;     // BSR matrix data structure

// =============================================================================
// LayerConfig: Configuration Parameters for a Single Layer
// =============================================================================
/**
 * Contains all parameters needed to execute one layer on the accelerator.
 * This includes matrix dimensions, quantization scales, memory addresses,
 * and operational flags.
 *
 * GEMM Operation: C[M×N] = A[M×K] × B[K×N]
 *   Where:
 *     A = Activations (INT8, row-major)
 *     B = Weights (INT8, BSR sparse or dense)
 *     C = Output (INT32 before re-quantization)
 *
 * For Convolution (im2col form):
 *     M = output_h × output_w (spatial output size)
 *     N = out_channels
 *     K = in_channels × kernel_h × kernel_w
 *
 * For Fully Connected:
 *     M = batch_size (typically 1 for inference)
 *     N = out_features
 *     K = in_features
 */
struct LayerConfig {
    // -------------------------------------------------------------------------
    // Matrix Dimensions
    // -------------------------------------------------------------------------
    uint32_t M;             ///< Output rows (batch × spatial or batch)
    uint32_t N;             ///< Output columns (out_channels or out_features)
    uint32_t K;             ///< Inner/reduction dimension
    
    // -------------------------------------------------------------------------
    // Tile Dimensions for Systolic Array
    // -------------------------------------------------------------------------
    // Tiling breaks large matrices into chunks that fit the 14×14 PE array.
    // Default values match systolic array size for optimal utilization.
    uint32_t Tm;            ///< Tile size for M dimension (default: 14)
    uint32_t Tn;            ///< Tile size for N dimension (default: 14)
    uint32_t Tk;            ///< Tile size for K dimension (default: 14 for BSR)
    
    // -------------------------------------------------------------------------
    // Quantization Parameters
    // -------------------------------------------------------------------------
    // INT32 output is dequantized: float_out = int32_out × scale_act × scale_wgt
    float scale_activation; ///< Activation quantization scale (from calibration)
    float scale_weight;     ///< Weight quantization scale (from training)
    
    // -------------------------------------------------------------------------
    // BSR Sparse Parameters
    // -------------------------------------------------------------------------
    // For sparse layers, weights are stored in Block Sparse Row format.
    // block_size = 14 (matches systolic array dimensions)
    uint32_t bsr_nnz_blocks;      ///< Number of non-zero 14×14 blocks
    uint32_t bsr_num_block_rows;  ///< Number of block rows (M / 14)
    uint32_t bsr_num_block_cols;  ///< Number of block columns (K / 14)
    
    // -------------------------------------------------------------------------
    // Physical Memory Addresses (DDR)
    // -------------------------------------------------------------------------
    // These addresses must be allocated via MemoryManager and be accessible
    // by the accelerator's DMA engines.
    uint64_t act_addr;      ///< Activation buffer physical address
    uint64_t wgt_addr;      ///< Weight buffer physical address (BSR or dense)
    uint64_t out_addr;      ///< Output buffer physical address
    uint64_t bsr_ptr_addr;  ///< BSR row_ptr array physical address
    uint64_t bsr_idx_addr;  ///< BSR col_idx array physical address
    
    // -------------------------------------------------------------------------
    // Operational Flags
    // -------------------------------------------------------------------------
    bool is_sparse;         ///< Use BSR sparse format (vs dense)
    bool has_relu;          ///< Apply ReLU activation to output
    bool has_bias;          ///< Add bias before activation (TODO: implement)
    
    /// Default constructor: Safe defaults for 14×14 systolic array
    LayerConfig()
        : M(0), N(0), K(0)
        , Tm(csr::SYSTOLIC_ROWS), Tn(csr::SYSTOLIC_COLS), Tk(csr::BLOCK_SIZE)
        , scale_activation(1.0f), scale_weight(1.0f)
        , bsr_nnz_blocks(0), bsr_num_block_rows(0), bsr_num_block_cols(0)
        , act_addr(0), wgt_addr(0), out_addr(0)
        , bsr_ptr_addr(0), bsr_idx_addr(0)
        , is_sparse(false), has_relu(true), has_bias(true)
    {}
    
    // -------------------------------------------------------------------------
    // Helper Methods
    // -------------------------------------------------------------------------
    
    /// Size of activation buffer in bytes (M × K INT8 values)
    size_t activation_size_bytes() const { return M * K * sizeof(int8_t); }
    
    /// Size of dense weight buffer in bytes (K × N INT8 values)
    size_t weight_size_bytes_dense() const { return K * N * sizeof(int8_t); }
    
    /// Size of output buffer in bytes (M × N INT32 values)
    size_t output_size_bytes() const { return M * N * sizeof(int32_t); }
    
    /// Number of tiles in M dimension (ceiling division)
    size_t num_tiles_m() const { return (M + Tm - 1) / Tm; }
    
    /// Number of tiles in N dimension
    size_t num_tiles_n() const { return (N + Tn - 1) / Tn; }
    
    /// Number of tiles in K dimension
    size_t num_tiles_k() const { return (K + Tk - 1) / Tk; }
    
    /// Total number of tiles to process
    size_t total_tiles() const { return num_tiles_m() * num_tiles_n() * num_tiles_k(); }
};

// =============================================================================
// PerfCounters: Performance Monitoring Data from Hardware Counters
// =============================================================================
/**
 * Container for hardware performance counters read from the perf.sv module.
 * These counters are captured (latched) when the accelerator signals 'done',
 * providing a snapshot of the last execution's performance.
 *
 * Counter Relationships:
 *   total_cycles = active_cycles + idle_cycles
 *
 * Performance Analysis:
 *   - utilization = active_cycles / total_cycles
 *     Ideal: >0.85 (85%), meaning PEs are busy most of the time
 *     Low utilization indicates memory-bound operation
 *
 *   - cache_hit_rate = cache_hits / (cache_hits + cache_misses)
 *     Ideal: >0.95 for repeated access patterns
 *     Low hit rate indicates working set exceeds cache capacity (64 entries)
 *
 *   - GOPS = total_ops / (total_cycles / clock_freq)
 *     For 14×14 array @ 100MHz: theoretical peak = 2×196×100M = 39.2 GOPS
 *     (2 ops per MAC: multiply + accumulate)
 */
struct PerfCounters {
    // -------------------------------------------------------------------------
    // Raw Counter Values (from perf.sv PERF_* registers)
    // -------------------------------------------------------------------------
    uint32_t total_cycles;   ///< Total execution cycles (start to done)
    uint32_t active_cycles;  ///< Cycles with valid data flowing through array
    uint32_t idle_cycles;    ///< Stall cycles (DMA wait, pipeline bubbles)
    uint32_t cache_hits;     ///< Metadata cache hits (meta_decode.sv)
    uint32_t cache_misses;   ///< Metadata cache misses (require DDR fetch)
    uint32_t decode_count;   ///< Total BSR decode operations performed
    
    // -------------------------------------------------------------------------
    // Derived Metrics
    // -------------------------------------------------------------------------
    
    /**
     * PE Array Utilization (0.0 to 1.0)
     * 
     * Measures what fraction of time the systolic array was actively computing.
     * Low utilization (<0.5) typically indicates:
     *   - Memory bandwidth bottleneck (DMA can't keep up)
     *   - Poor data locality (many cache misses)
     *   - Small layer dimensions (not enough work to fill array)
     *
     * @return Fraction of time spent in active computation
     */
    float utilization() const {
        if (total_cycles == 0) return 0.0f;
        return static_cast<float>(active_cycles) / static_cast<float>(total_cycles);
    }
    
    /**
     * Stall Percentage (0.0 to 100.0)
     * 
     * Inverse of utilization, expressed as percentage.
     * High stall percent (>30%) indicates optimization opportunities:
     *   - Consider prefetching activations
     *   - Increase cache depth if hit rate is low
     *   - Review BSR block density
     *
     * @return Percentage of time spent stalled
     */
    float stall_percent() const {
        if (total_cycles == 0) return 0.0f;
        return static_cast<float>(idle_cycles) / static_cast<float>(total_cycles) * 100.0f;
    }
    
    /**
     * Metadata Cache Hit Rate (0.0 to 1.0)
     * 
     * BSR metadata (row_ptr, col_idx) is cached in meta_decode.sv.
     * Cache is 64-entry direct-mapped with 6-bit index (row_idx[5:0]).
     *
     * Low hit rate indicates:
     *   - Working set too large (>64 unique block rows accessed)
     *   - Poor spatial locality in BSR access pattern
     *   - May need larger cache or different indexing strategy
     *
     * @return Cache hit ratio (1.0 = perfect, 0.0 = no hits)
     */
    float cache_hit_rate() const {
        uint32_t total = cache_hits + cache_misses;
        if (total == 0) return 0.0f;
        return static_cast<float>(cache_hits) / static_cast<float>(total);
    }
    
    /**
     * Compute GOPS (Giga Operations Per Second)
     * 
     * Calculation:
     *   seconds = total_cycles / (clock_mhz × 10^6)
     *   GOPS = total_ops / (seconds × 10^9)
     *
     * For GEMM: total_ops = 2 × M × N × K (multiply + accumulate per element)
     * 
     * Reference (14×14 array @ 100MHz):
     *   Peak = 2 × 196 × 100×10^6 = 39.2 GOPS
     *   Typical achievable: 25-35 GOPS (depending on sparsity)
     *
     * @param clock_mhz Operating clock frequency in MHz (default: 100)
     * @param total_ops Total operations in the workload (2×M×N×K for GEMM)
     * @return Throughput in giga-operations per second
     */
    float gops(float clock_mhz, uint64_t total_ops) const {
        if (total_cycles == 0) return 0.0f;
        float seconds = static_cast<float>(total_cycles) / (clock_mhz * 1e6f);
        return static_cast<float>(total_ops) / (seconds * 1e9f);
    }
    
    /// Reset all counters to zero
    void clear() {
        total_cycles = active_cycles = idle_cycles = 0;
        cache_hits = cache_misses = decode_count = 0;
    }
};

// =============================================================================
// AcceleratorError: Exception Class for Hardware/Driver Errors
// =============================================================================
/**
 * Typed exception for accelerator operations.
 * 
 * Error Codes:
 *   INIT_FAILED    - Failed to memory-map AXI registers or detect hardware
 *   TIMEOUT        - wait_done() exceeded timeout_ms without completion
 *   DMA_ERROR      - DMA transfer failed (AXI SLVERR/DECERR response)
 *   ILLEGAL_COMMAND- Invalid command sequence (e.g., start while busy)
 *   INVALID_CONFIG - LayerConfig validation failed (dimensions, addresses)
 *   MEMORY_ERROR   - Memory allocation failed or invalid physical address
 *
 * Example Usage:
 *   try {
 *       driver.wait_done(1000);  // 1 second timeout
 *   } catch (const AcceleratorError& e) {
 *       if (e.code() == AcceleratorError::Code::TIMEOUT) {
 *           driver.reset();  // Attempt recovery
 *       }
 *   }
 */
class AcceleratorError : public std::runtime_error {
public:
    enum class Code {
        INIT_FAILED,      ///< Hardware initialization failed
        TIMEOUT,          ///< Operation exceeded timeout
        DMA_ERROR,        ///< DMA transfer error (AXI error response)
        ILLEGAL_COMMAND,  ///< Invalid command for current state
        INVALID_CONFIG,   ///< LayerConfig validation failed
        MEMORY_ERROR      ///< Memory allocation or mapping error
    };
    
    AcceleratorError(Code code, const std::string& message)
        : std::runtime_error(message), code_(code) {}
    
    Code code() const { return code_; }
    
private:
    Code code_;
};

// =============================================================================
// AcceleratorDriver: Main Host-Side Driver Class
// =============================================================================
/**
 * High-level driver for the INT8 sparse systolic array accelerator.
 * 
 * This class provides a unified interface for:
 *   1. Hardware initialization and reset
 *   2. Layer configuration (dimensions, addresses, scales)
 *   3. DMA buffer management
 *   4. Execution control (start, wait, abort)
 *   5. Performance monitoring
 *
 * Operating Modes:
 *   FPGA           - Direct hardware access via memory-mapped AXI
 *   SIMULATION     - Verilator/cocotb simulation via VPI or socket
 *   SOFTWARE_MODEL - Pure C++ behavioral model for debugging
 *
 * Execution Flow:
 *   ┌──────────────────────────────────────────────────────────────────┐
 *   │  1. AcceleratorDriver driver(Mode::FPGA);                        │
 *   │  2. driver.initialize();                                         │
 *   │  3. driver.load_weights_bsr("model_weights/");                   │
 *   │  4. for each layer:                                              │
 *   │       driver.configure_layer(config);                            │
 *   │       driver.set_activation_buffer(act_addr, size);              │
 *   │       driver.start();                                            │
 *   │       driver.wait_done(5000);  // 5 second timeout               │
 *   │       auto perf = driver.read_perf_counters();                   │
 *   └──────────────────────────────────────────────────────────────────┘
 *
 * Memory Ownership:
 *   - MemoryManager handles DDR allocation (CMA on Zynq)
 *   - User provides LayerConfig with physical addresses
 *   - Driver does NOT free user-provided buffers
 *
 * Thread Safety:
 *   - NOT thread-safe. Use external synchronization if needed.
 *   - Typical use: single-threaded inference loop
 *
 * Base Address Mapping (Zynq-7020):
 *   - ACCEL_BASE_ADDR = 0x43C00000 (AXI GP0 slave)
 *   - DDR base for DMA = 0x00000000 (HP0 master)
 *
 * Magic Numbers Explained:
 *   - 5000 (timeout_ms): Default wait_done timeout (5 seconds)
 *     Typical layer execution: 10-100ms, 5s allows for debug/slow sim
 *   - 1000 (dma_timeout_ms): Default DMA wait timeout (1 second)
 *     DMA transfers typically complete in <1ms for reasonable sizes
 */
class AcceleratorDriver {
public:
    /// Operating mode for the driver
    enum class Mode {
        FPGA,             ///< Real hardware via /dev/mem mmap
        SIMULATION,       ///< RTL simulation via Verilator/cocotb
        SOFTWARE_MODEL    ///< Pure C++ behavioral model
    };
    
    // =========================================================================
    // Construction & Initialization
    // =========================================================================
    
    /**
     * Construct driver for specified mode.
     * 
     * @param mode Operating mode (FPGA, SIMULATION, or SOFTWARE_MODEL)
     * @param base_addr AXI-Lite base address (default: 0x43C00000 for Zynq GP0)
     */
    explicit AcceleratorDriver(Mode mode, uint64_t base_addr = csr::ACCEL_BASE_ADDR);
    ~AcceleratorDriver();
    
    // Non-copyable, movable
    AcceleratorDriver(const AcceleratorDriver&) = delete;
    AcceleratorDriver& operator=(const AcceleratorDriver&) = delete;
    AcceleratorDriver(AcceleratorDriver&&) noexcept;
    AcceleratorDriver& operator=(AcceleratorDriver&&) noexcept;
    
    /// Initialize hardware: map registers, reset accelerator, verify version
    void initialize();
    
    /// Soft reset: stops all operations, clears FIFOs, resets state machines
    void reset();
    
    /// Check if initialize() has been called successfully
    bool is_initialized() const { return initialized_; }
    
    /// Read hardware version register (for compatibility checking)
    uint32_t get_version();
    
    // =========================================================================
    // Layer Configuration
    // =========================================================================
    
    /// Configure accelerator for a layer (writes all CSR registers)
    void configure_layer(const LayerConfig& config);
    
    /// Set GEMM dimensions M×N×K
    void set_dimensions(uint32_t M, uint32_t N, uint32_t K);
    
    /// Set tile dimensions for loop blocking
    void set_tile_dimensions(uint32_t Tm, uint32_t Tn, uint32_t Tk);
    
    /// Set quantization scales (stored as IEEE 754 bits)
    void set_scales(float scale_act, float scale_wgt);
    
    // =========================================================================
    // DMA Buffer Configuration
    // =========================================================================
    
    /// Set activation input buffer (physical DDR address)
    void set_activation_buffer(uint64_t phys_addr, uint32_t size_bytes);
    
    /// Set weight buffer (physical DDR address, dense or BSR values)
    void set_weight_buffer(uint64_t phys_addr, uint32_t size_bytes);
    
    /// Set output buffer (physical DDR address, must hold M×N×4 bytes)
    void set_output_buffer(uint64_t phys_addr);
    
    /// Set BSR metadata buffers (row_ptr and col_idx arrays)
    void set_bsr_buffers(uint64_t ptr_addr, uint64_t idx_addr, uint32_t nnz_blocks);
    
    // =========================================================================
    // Execution Control
    // =========================================================================
    
    /// Start layer execution (writes START bit to CTRL register)
    void start();
    
    /// Emergency stop (may leave hardware in inconsistent state)
    void abort();
    
    /**
     * Block until execution completes or timeout.
     * 
     * @param timeout_ms Maximum wait time in milliseconds (default: 5000)
     * @throws AcceleratorError(TIMEOUT) if done not signaled within timeout
     */
    void wait_done(uint32_t timeout_ms = 5000);
    
    /// High-level: configure + start + wait + return results
    void run_layer(const LayerConfig& config, const int8_t* input, int32_t* output);
    
    /// High-level: run a previously configured layer by index
    void run_layer(size_t layer_idx, const int8_t* input, int32_t* output);
    
    // =========================================================================
    // Status Checking
    // =========================================================================
    
    /// Check if accelerator is currently executing
    bool is_busy();
    
    /// Check if execution completed successfully
    bool is_done();
    
    /// Check if any error flags are set
    bool has_error();
    
    /// Read raw STATUS register value
    uint32_t read_status();
    
    /// Clear error flags (write-1-to-clear)
    void clear_errors();
    
    /// Print current status to stderr (for debugging)
    void dump_status();
    
    // =========================================================================
    // Performance Monitoring
    // =========================================================================
    
    /// Read all performance counters (latched on done)
    PerfCounters read_perf_counters();
    
    /// Reset performance counters for new measurement
    void reset_perf_counters();
    
    // =========================================================================
    // Weight Management
    // =========================================================================
    
    /// Load BSR weights from directory (expects layer_N_values.bin, etc.)
    void load_weights_bsr(const std::string& weight_dir);
    
    /// Set BSR weights for a specific layer
    void set_layer_weights(size_t layer_idx, const BSRMatrix& bsr);
    
    /// Set raw BSR weight data for a layer
    void set_layer_weights(size_t layer_idx, const void* bsr_data, size_t size);
    
    // =========================================================================
    // DMA Control
    // =========================================================================
    
    /// Initiate weight DMA transfer from DDR
    void start_weight_dma(uint64_t src_addr, uint32_t len);
    
    /// Initiate activation DMA transfer from DDR
    void start_activation_dma(uint64_t src_addr, uint32_t len);
    
    /// Wait for DMA transfer to complete
    void wait_dma_done(uint32_t timeout_ms = 1000);
    
    /// Check if any DMA transfer is in progress
    bool is_dma_busy();
    
    // =========================================================================
    // Direct Register Access (for debugging/testing)
    // =========================================================================
    
    /// Write 32-bit value to CSR at offset
    void write_reg(uint32_t offset, uint32_t value);
    
    /// Read 32-bit value from CSR at offset
    uint32_t read_reg(uint32_t offset);
    
    // =========================================================================
    // Accessors
    // =========================================================================
    
    Mode mode() const { return mode_; }
    AXIMaster* axi() { return axi_.get(); }
    MemoryManager* memory() { return memory_.get(); }
    
private:
    // -------------------------------------------------------------------------
    // Private Member Variables
    // -------------------------------------------------------------------------
    Mode mode_;                                 ///< Operating mode
    uint64_t base_addr_;                        ///< AXI-Lite base address
    bool initialized_;                          ///< Has initialize() been called?
    
    std::unique_ptr<AXIMaster> axi_;            ///< AXI backend for reg access
    std::unique_ptr<MemoryManager> memory_;     ///< DDR memory allocator
    std::unique_ptr<BSRPacker> bsr_packer_;     ///< BSR format converter
    
    std::vector<LayerConfig> layer_configs_;    ///< Per-layer configurations
    std::vector<std::vector<uint8_t>> layer_weights_;  ///< Packed BSR weights
    size_t current_layer_;                      ///< Index of currently configured layer
    
    // -------------------------------------------------------------------------
    // Private Helper Methods
    // -------------------------------------------------------------------------
    
    /// Convert float to 32-bit representation (for scale registers)
    static uint32_t float_to_bits(float f) {
        union { float f; uint32_t u; } converter;
        converter.f = f;
        return converter.u;
    }
    
    /// Convert 32-bit representation back to float
    static float bits_to_float(uint32_t u) {
        union { float f; uint32_t u; } converter;
        converter.u = u;
        return converter.f;
    }
    
    /// Validate LayerConfig before use (throws on error)
    void validate_config(const LayerConfig& config);
    
    /// Create appropriate AXI backend for current mode
    std::unique_ptr<AXIBackend> create_backend();
};

} // namespace resnet_accel

#endif // ACCELERATOR_DRIVER_HPP
