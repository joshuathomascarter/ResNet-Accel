/**
 * @file performance_counters.hpp
 * @brief Hardware performance counter reading and analysis for optimization
 * @author ResNet-Accel Team
 * @date 2024
 * @copyright MIT License
 * 
 * @details
 * Reads and interprets hardware performance counters from the accelerator.
 * Provides metrics for:
 * - Utilization and efficiency analysis
 * - Throughput measurement (GOPS)
 * - Stall breakdown for bottleneck identification
 * - Roofline model operational intensity
 * 
 * ## Hardware Counters (from hw/rtl/csr.sv):
 * | Register | Offset | Description |
 * |----------|--------|-------------|
 * | PERF_TOTAL | 0x40 | Total cycles from start to done |
 * | PERF_ACTIVE | 0x44 | Cycles where busy was high |
 * | PERF_IDLE | 0x48 | Cycles where busy was low |
 * | PERF_CACHE_HITS | 0x4C | Metadata cache hits |
 * | PERF_CACHE_MISSES | 0x50 | Metadata cache misses |
 * | PERF_DECODE_COUNT | 0x54 | Metadata decode operations |
 * 
 * ## Derived Metrics:
 * - **Utilization** = active_cycles / total_cycles (0.0-1.0)
 * - **PE Efficiency** = mac_ops / (active_cycles * N_PES)
 * - **Throughput** = mac_ops * 2 / (total_cycles / clock_freq) GOPS
 * - **Cache Hit Rate** = cache_hits / (cache_hits + cache_misses)
 */

#ifndef PERFORMANCE_COUNTERS_HPP
#define PERFORMANCE_COUNTERS_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <iosfwd>

namespace resnet_accel {

// Forward declarations
class AXIMaster;

/**
 * @brief Raw counter values read from hardware
 */
struct RawCounters {
    std::uint32_t total_cycles = 0;       ///< Total cycles from start to done
    std::uint32_t active_cycles = 0;      ///< Cycles with busy high
    std::uint32_t idle_cycles = 0;        ///< Cycles with busy low
    std::uint32_t cache_hits = 0;         ///< Metadata cache hits
    std::uint32_t cache_misses = 0;       ///< Metadata cache misses
    std::uint32_t decode_count = 0;       ///< Metadata decode operations
    
    /** @brief Subtract start counters to get delta */
    RawCounters operator-(const RawCounters& start) const {
        return {
            total_cycles - start.total_cycles,
            active_cycles - start.active_cycles,
            idle_cycles - start.idle_cycles,
            cache_hits - start.cache_hits,
            cache_misses - start.cache_misses,
            decode_count - start.decode_count
        };
    }
};

/**
 * @brief Computed performance metrics
 */
struct PerfMetrics {
    // Basic metrics
    double utilization = 0.0;           ///< Active / total (0.0-1.0)
    double pe_efficiency = 0.0;         ///< PE usage efficiency (0.0-1.0)
    double throughput_gops = 0.0;       ///< Giga-operations per second
    double latency_ms = 0.0;            ///< Milliseconds for operation
    
    // Cache metrics
    double cache_hit_rate = 0.0;        ///< Cache hits / total accesses
    
    // Bandwidth
    double bandwidth_gbps = 0.0;        ///< GB/s achieved
    double operational_intensity = 0.0; ///< FLOPS per byte
    
    // Stall analysis
    double stall_ratio = 0.0;           ///< Idle / total
    
    // Counts
    std::uint64_t mac_operations = 0;   ///< Estimated MAC ops
    std::uint64_t data_bytes = 0;       ///< Data transferred
};

/**
 * @brief Performance counter reader and analyzer
 * 
 * ## Example Usage
 * ```cpp
 * PerformanceCounters perf(&axi, 100.0);  // 100 MHz clock
 * 
 * perf.start_timing();
 * // ... run accelerator ...
 * auto metrics = perf.stop_timing(data_bytes);
 * 
 * std::cout << "Throughput: " << metrics.throughput_gops << " GOPS\n";
 * std::cout << "Utilization: " << (metrics.utilization * 100) << "%\n";
 * perf.print_report(metrics, std::cout);
 * ```
 */
class PerformanceCounters {
public:
    /**
     * @brief Construct performance counter interface
     * @param axi Pointer to AXI master for register access
     * @param clock_mhz Clock frequency in MHz (default 100.0)
     * @param num_pes Number of processing elements (default 256 for 16x16)
     */
    explicit PerformanceCounters(AXIMaster* axi, 
                                 double clock_mhz = 100.0,
                                 std::size_t num_pes = 256);
    
    //==========================================================================
    // Counter Reading
    //==========================================================================
    
    /**
     * @brief Read raw counters from hardware
     * @return Current counter values
     */
    [[nodiscard]] RawCounters read_raw() const;
    
    /**
     * @brief Reset all counters to zero
     * 
     * Counters typically auto-reset on START, but can be manually cleared.
     */
    void reset();
    
    //==========================================================================
    // Timing Measurement
    //==========================================================================
    
    /**
     * @brief Start timing measurement
     * 
     * Captures current counter state as baseline.
     */
    void start_timing();
    
    /**
     * @brief Stop timing and compute metrics
     * @param data_bytes Total bytes transferred (for bandwidth calc)
     * @param mac_ops Actual MAC operations (if known, else estimated)
     * @return Computed performance metrics
     */
    [[nodiscard]] PerfMetrics stop_timing(std::size_t data_bytes = 0,
                                          std::uint64_t mac_ops = 0);
    
    /**
     * @brief Read metrics without stopping timer
     * @param data_bytes Total bytes transferred
     * @param mac_ops Actual MAC operations
     * @return Current performance metrics
     */
    [[nodiscard]] PerfMetrics read_metrics(std::size_t data_bytes = 0,
                                           std::uint64_t mac_ops = 0) const;
    
    //==========================================================================
    // Metric Computation
    //==========================================================================
    
    /**
     * @brief Compute metrics from raw counters
     * @param raw Raw counter values
     * @param data_bytes Data transferred in bytes
     * @param mac_ops MAC operations performed
     * @return Computed metrics
     */
    [[nodiscard]] PerfMetrics compute_metrics(const RawCounters& raw,
                                              std::size_t data_bytes,
                                              std::uint64_t mac_ops) const;
    
    //==========================================================================
    // Reporting
    //==========================================================================
    
    /**
     * @brief Print formatted performance report
     * @param metrics Metrics to print
     * @param os Output stream
     */
    static void print_report(const PerfMetrics& metrics, std::ostream& os);
    
    /**
     * @brief Get human-readable summary string
     * @param metrics Metrics to summarize
     * @return Summary string
     */
    [[nodiscard]] static std::string summary(const PerfMetrics& metrics);
    
    //==========================================================================
    // Accessors
    //==========================================================================
    
    /** @brief Get clock frequency in MHz */
    [[nodiscard]] double get_clock_mhz() const { return clock_mhz_; }
    
    /** @brief Get number of PEs */
    [[nodiscard]] std::size_t get_num_pes() const { return num_pes_; }
    
    /** @brief Get theoretical peak GOPS */
    [[nodiscard]] double get_peak_gops() const {
        return (num_pes_ * 2.0 * clock_mhz_) / 1000.0;
    }

private:
    AXIMaster* axi_;
    double clock_mhz_;
    std::size_t num_pes_;
    RawCounters start_counters_;
    bool timing_active_ = false;
};

} // namespace resnet_accel

#endif // PERFORMANCE_COUNTERS_HPP
