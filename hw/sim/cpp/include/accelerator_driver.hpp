/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                       ACCELERATOR_DRIVER.HPP                              ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  REPLACES: sw/host/accel.py                                               ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  Top-level driver for the ResNet-18 sparse accelerator.                  ║
 * ║  Works with real FPGA (Zynq-7020) or Verilator simulation.               ║
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

// Forward declarations
class AXIMaster;
class AXIBackend;
class MemoryManager;
class BSRPacker;
struct BSRMatrix;

// =============================================================================
// LayerConfig: Configuration parameters for a single layer
// =============================================================================

struct LayerConfig {
    // Matrix dimensions for GEMM: C[M×N] = A[M×K] × B[K×N]
    uint32_t M;             // Output rows
    uint32_t N;             // Output columns
    uint32_t K;             // Inner dimension
    
    // Tile dimensions for systolic array
    uint32_t Tm;            // Tile size for M dimension
    uint32_t Tn;            // Tile size for N dimension
    uint32_t Tk;            // Tile size for K dimension
    
    // Quantization parameters
    float scale_activation;
    float scale_weight;
    
    // BSR sparse parameters
    uint32_t bsr_nnz_blocks;
    uint32_t bsr_num_block_rows;
    uint32_t bsr_num_block_cols;
    
    // Memory addresses (physical)
    uint64_t act_addr;
    uint64_t wgt_addr;
    uint64_t out_addr;
    uint64_t bsr_ptr_addr;
    uint64_t bsr_idx_addr;
    
    // Flags
    bool is_sparse;
    bool has_relu;
    bool has_bias;
    
    LayerConfig()
        : M(0), N(0), K(0)
        , Tm(csr::SYSTOLIC_ROWS), Tn(csr::SYSTOLIC_COLS), Tk(csr::BLOCK_SIZE)
        , scale_activation(1.0f), scale_weight(1.0f)
        , bsr_nnz_blocks(0), bsr_num_block_rows(0), bsr_num_block_cols(0)
        , act_addr(0), wgt_addr(0), out_addr(0)
        , bsr_ptr_addr(0), bsr_idx_addr(0)
        , is_sparse(false), has_relu(true), has_bias(true)
    {}
    
    size_t activation_size_bytes() const { return M * K * sizeof(int8_t); }
    size_t weight_size_bytes_dense() const { return K * N * sizeof(int8_t); }
    size_t output_size_bytes() const { return M * N * sizeof(int32_t); }
    size_t num_tiles_m() const { return (M + Tm - 1) / Tm; }
    size_t num_tiles_n() const { return (N + Tn - 1) / Tn; }
    size_t num_tiles_k() const { return (K + Tk - 1) / Tk; }
    size_t total_tiles() const { return num_tiles_m() * num_tiles_n() * num_tiles_k(); }
};

// =============================================================================
// PerfCounters: Performance monitoring data
// =============================================================================

struct PerfCounters {
    uint32_t total_cycles;
    uint32_t active_cycles;
    uint32_t idle_cycles;
    uint32_t cache_hits;
    uint32_t cache_misses;
    uint32_t decode_count;
    
    float utilization() const {
        if (total_cycles == 0) return 0.0f;
        return static_cast<float>(active_cycles) / static_cast<float>(total_cycles);
    }
    
    float stall_percent() const {
        if (total_cycles == 0) return 0.0f;
        return static_cast<float>(idle_cycles) / static_cast<float>(total_cycles) * 100.0f;
    }
    
    float cache_hit_rate() const {
        uint32_t total = cache_hits + cache_misses;
        if (total == 0) return 0.0f;
        return static_cast<float>(cache_hits) / static_cast<float>(total);
    }
    
    float gops(float clock_mhz, uint64_t total_ops) const {
        if (total_cycles == 0) return 0.0f;
        float seconds = static_cast<float>(total_cycles) / (clock_mhz * 1e6f);
        return static_cast<float>(total_ops) / (seconds * 1e9f);
    }
    
    void clear() {
        total_cycles = active_cycles = idle_cycles = 0;
        cache_hits = cache_misses = decode_count = 0;
    }
};

// =============================================================================
// AcceleratorError: Exception class
// =============================================================================

class AcceleratorError : public std::runtime_error {
public:
    enum class Code {
        INIT_FAILED,
        TIMEOUT,
        DMA_ERROR,
        ILLEGAL_COMMAND,
        INVALID_CONFIG,
        MEMORY_ERROR
    };
    
    AcceleratorError(Code code, const std::string& message)
        : std::runtime_error(message), code_(code) {}
    
    Code code() const { return code_; }
    
private:
    Code code_;
};

// =============================================================================
// AcceleratorDriver: Main driver class
// =============================================================================

class AcceleratorDriver {
public:
    enum class Mode {
        FPGA,
        SIMULATION,
        SOFTWARE_MODEL
    };
    
    // Construction
    explicit AcceleratorDriver(Mode mode, uint64_t base_addr = csr::ACCEL_BASE_ADDR);
    ~AcceleratorDriver();
    
    AcceleratorDriver(const AcceleratorDriver&) = delete;
    AcceleratorDriver& operator=(const AcceleratorDriver&) = delete;
    AcceleratorDriver(AcceleratorDriver&&) noexcept;
    AcceleratorDriver& operator=(AcceleratorDriver&&) noexcept;
    
    // Initialization
    void initialize();
    void reset();
    bool is_initialized() const { return initialized_; }
    uint32_t get_version();
    
    // Layer Configuration
    void configure_layer(const LayerConfig& config);
    void set_dimensions(uint32_t M, uint32_t N, uint32_t K);
    void set_tile_dimensions(uint32_t Tm, uint32_t Tn, uint32_t Tk);
    void set_scales(float scale_act, float scale_wgt);
    
    // DMA Buffer Configuration
    void set_activation_buffer(uint64_t phys_addr, uint32_t size_bytes);
    void set_weight_buffer(uint64_t phys_addr, uint32_t size_bytes);
    void set_output_buffer(uint64_t phys_addr);
    void set_bsr_buffers(uint64_t ptr_addr, uint64_t idx_addr, uint32_t nnz_blocks);
    
    // Execution Control
    void start();
    void abort();
    void wait_done(uint32_t timeout_ms = 5000);
    void run_layer(const LayerConfig& config, const int8_t* input, int32_t* output);
    void run_layer(size_t layer_idx, const int8_t* input, int32_t* output);
    
    // Status Checking
    bool is_busy();
    bool is_done();
    bool has_error();
    uint32_t read_status();
    void clear_errors();
    void dump_status();
    
    // Performance Monitoring
    PerfCounters read_perf_counters();
    void reset_perf_counters();
    
    // Weight Management
    void load_weights_bsr(const std::string& weight_dir);
    void set_layer_weights(size_t layer_idx, const BSRMatrix& bsr);
    void set_layer_weights(size_t layer_idx, const void* bsr_data, size_t size);
    
    // DMA Control
    void start_weight_dma(uint64_t src_addr, uint32_t len);
    void start_activation_dma(uint64_t src_addr, uint32_t len);
    void wait_dma_done(uint32_t timeout_ms = 1000);
    bool is_dma_busy();
    
    // Direct Register Access
    void write_reg(uint32_t offset, uint32_t value);
    uint32_t read_reg(uint32_t offset);
    
    // Accessors
    Mode mode() const { return mode_; }
    AXIMaster* axi() { return axi_.get(); }
    MemoryManager* memory() { return memory_.get(); }
    
private:
    Mode mode_;
    uint64_t base_addr_;
    bool initialized_;
    
    std::unique_ptr<AXIMaster> axi_;
    std::unique_ptr<MemoryManager> memory_;
    std::unique_ptr<BSRPacker> bsr_packer_;
    
    std::vector<LayerConfig> layer_configs_;
    std::vector<std::vector<uint8_t>> layer_weights_;
    size_t current_layer_;
    
    static uint32_t float_to_bits(float f) {
        union { float f; uint32_t u; } converter;
        converter.f = f;
        return converter.u;
    }
    
    static float bits_to_float(uint32_t u) {
        union { float f; uint32_t u; } converter;
        converter.u = u;
        return converter.f;
    }
    
    void validate_config(const LayerConfig& config);
    std::unique_ptr<AXIBackend> create_backend();
};

} // namespace resnet_accel

#endif // ACCELERATOR_DRIVER_HPP
