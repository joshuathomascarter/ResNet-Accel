/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                    PERFORMANCE_COUNTERS.HPP                               ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  REPLACES: Performance monitoring code scattered in sw/host/accel.py     ║
 * ║            No direct Python equivalent - this is enhanced functionality  ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║    Read and interpret hardware performance counters from the             ║
 * ║    accelerator. Provides metrics for optimization and debugging:         ║
 * ║    utilization, throughput, stall analysis, roofline model data.         ║
 * ║                                                                           ║
 * ║  WHY THIS MATTERS FOR HACKATHONS:                                        ║
 * ║    • Judges LOVE seeing actual performance numbers                       ║
 * ║    • Shows you understand hardware efficiency                            ║
 * ║    • Proves your accelerator actually provides speedup                   ║
 * ║    • Enables data-driven optimization                                    ║
 * ║                                                                           ║
 * ║  HARDWARE COUNTERS (from hw/rtl/csr.sv):                                 ║
 * ║    Register     Offset   Description                                     ║
 * ║    ─────────────────────────────────────────────────────────────────     ║
 * ║    CYCLE_COUNT   0x50    Total clock cycles since start                  ║
 * ║    COMPUTE_CYC   0x54    Cycles spent computing (not stalled)            ║
 * ║    STALL_CYC     0x58    Cycles spent stalled                            ║
 * ║    MAC_OPS       0x5C    Total MAC operations performed                  ║
 * ║                                                                           ║
 * ║  DERIVED METRICS:                                                         ║
 * ║                                                                           ║
 * ║    Utilization = compute_cycles / total_cycles                           ║
 * ║      - 100% = no stalls, perfect efficiency                              ║
 * ║      - <50% = something is wrong (memory bottleneck?)                    ║
 * ║                                                                           ║
 * ║    PE Efficiency = mac_ops / (compute_cycles * N_PES)                    ║
 * ║      - How well we're using the 256 PEs                                  ║
 * ║      - <100% means some PEs idle during compute                          ║
 * ║                                                                           ║
 * ║    Throughput (GOPS) = mac_ops * 2 / (total_cycles / clock_freq)         ║
 * ║      - *2 because each MAC is 2 operations (multiply + add)              ║
 * ║      - Compare to theoretical peak: 16*16*2*100MHz = 51.2 GOPS           ║
 * ║                                                                           ║
 * ║    Latency = total_cycles / clock_freq                                   ║
 * ║      - Time per inference in seconds/milliseconds                        ║
 * ║                                                                           ║
 * ║    Memory Bandwidth = data_transferred / latency                         ║
 * ║      - Compare to DDR bandwidth limit                                    ║
 * ║                                                                           ║
 * ║  STALL BREAKDOWN:                                                         ║
 * ║    The accelerator stalls when:                                          ║
 * ║      • Activation buffer empty (waiting for input DMA)                   ║
 * ║      • Weight buffer empty (waiting for weight DMA)                      ║
 * ║      • Output buffer full (waiting for output DMA)                       ║
 * ║      • BSR scheduler fetching next block metadata                        ║
 * ║                                                                           ║
 * ║  ROOFLINE MODEL:                                                          ║
 * ║    Plot operational intensity vs achieved GOPS to see if                 ║
 * ║    compute-bound or memory-bound:                                        ║
 * ║                                                                           ║
 * ║         GOPS │        ╱ Peak (51.2 GOPS)                                 ║
 * ║              │      ╱                                                    ║
 * ║              │    ╱   ● Dense                                            ║
 * ║              │  ╱        ● Sparse (higher intensity, same GOPS)          ║
 * ║              │╱____________________________________________              ║
 * ║              0           Operational Intensity (GOPS/GB)                 ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#ifndef PERFORMANCE_COUNTERS_HPP
#define PERFORMANCE_COUNTERS_HPP

#include <cstdint>
#include <string>

// Forward declaration
class AXIMaster;

/**
 * Raw counter values read from hardware
 */
struct RawCounters {
    uint32_t total_cycles;      // CYCLE_COUNT register
    uint32_t compute_cycles;    // COMPUTE_CYC register
    uint32_t stall_cycles;      // STALL_CYC register
    uint32_t mac_operations;    // MAC_OPS register
    
    // Optional detailed stall breakdown (if hardware supports)
    uint32_t act_stall_cycles;  // Stalled on activation buffer
    uint32_t wgt_stall_cycles;  // Stalled on weight buffer
    uint32_t out_stall_cycles;  // Stalled on output buffer
};

/**
 * Computed performance metrics
 */
struct PerfMetrics {
    // Basic metrics
    double utilization;          // compute_cycles / total_cycles (0.0 to 1.0)
    double pe_efficiency;        // How well PEs are used (0.0 to 1.0)
    double throughput_gops;      // Giga-operations per second
    double latency_ms;           // Milliseconds for this operation
    
    // Detailed breakdown
    double stall_ratio;          // stall_cycles / total_cycles
    double act_stall_ratio;      // Activation buffer stall ratio
    double wgt_stall_ratio;      // Weight buffer stall ratio
    double out_stall_ratio;      // Output buffer stall ratio
    
    // Bandwidth
    double achieved_bandwidth_gbps;  // GB/s actually achieved
    
    // Roofline data
    double operational_intensity;    // FLOPS per byte of memory traffic
};

/**
 * Performance counter reader and analyzer
 */
class PerformanceCounters {
public:
    /**
     * Constructor
     * 
     * @param axi         AXI master for register access
     * @param clock_mhz   Clock frequency in MHz (default 100)
     */
    explicit PerformanceCounters(AXIMaster* axi, double clock_mhz = 100.0);
    
    /**
     * Read raw counters from hardware
     * 
     * TODO: Implement - read from register offsets 0x50-0x5C
     */
    RawCounters read_raw();
    
    /**
     * Reset all counters to zero
     * 
     * TODO: Implement - write to counter reset register
     */
    void reset();
    
    /**
     * Read and compute metrics
     * 
     * @param data_bytes  Bytes transferred for bandwidth calculation
     */
    PerfMetrics read_metrics(size_t data_bytes = 0);
    
    /**
     * Start timing (read initial counter values)
     */
    void start_timing();
    
    /**
     * Stop timing and compute metrics
     * 
     * @param data_bytes  Bytes transferred for bandwidth calculation
     */
    PerfMetrics stop_timing(size_t data_bytes = 0);
    
    /**
     * Print formatted performance report
     */
    static void print_report(const PerfMetrics& metrics);
    
    /**
     * Print comparison between two runs (e.g., dense vs sparse)
     */
    static void print_comparison(const PerfMetrics& dense, 
                                  const PerfMetrics& sparse);
    
    /**
     * Get theoretical peak performance
     */
    double get_peak_gops() const;
    
    /**
     * Generate CSV line for logging
     */
    static std::string to_csv(const PerfMetrics& metrics);
    
    /**
     * CSV header for logging
     */
    static std::string csv_header();
    
private:
    AXIMaster* axi_;
    double clock_mhz_;
    RawCounters start_counters_;
    
    // Hardware constants
    static constexpr size_t N_PES = 256;  // 16x16 array
    
    /**
     * Compute metrics from raw counters
     */
    PerfMetrics compute_metrics(const RawCounters& raw, size_t data_bytes);
};

/**
 * Layer-level performance tracking
 * Aggregates metrics across an entire inference
 */
class InferenceProfiler {
public:
    InferenceProfiler();
    
    /**
     * Record metrics for one layer
     */
    void record_layer(const std::string& layer_name, const PerfMetrics& metrics);
    
    /**
     * Get total inference time
     */
    double total_latency_ms() const;
    
    /**
     * Get average utilization across layers
     */
    double average_utilization() const;
    
    /**
     * Get total GOPS across inference
     */
    double total_gops() const;
    
    /**
     * Print per-layer breakdown
     */
    void print_layer_breakdown() const;
    
    /**
     * Find bottleneck layer
     */
    std::string find_bottleneck() const;
    
    /**
     * Export to JSON for visualization
     */
    std::string export_json() const;
    
    /**
     * Clear recorded data
     */
    void reset();
    
private:
    struct LayerData {
        std::string name;
        PerfMetrics metrics;
    };
    std::vector<LayerData> layers_;
};

#endif // PERFORMANCE_COUNTERS_HPP
