/**
 * @file performance_counters.cpp
 * @brief Implementation of hardware performance counter reading and analysis
 */

#include "performance_counters.hpp"
#include "axi_master.hpp"
#include "csr_map.hpp"

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace resnet_accel {

//==============================================================================
// Constructor
//==============================================================================

PerformanceCounters::PerformanceCounters(AXIMaster* axi, 
                                         double clock_mhz,
                                         std::size_t num_pes)
    : axi_(axi)
    , clock_mhz_(clock_mhz)
    , num_pes_(num_pes)
    , start_counters_{}
    , timing_active_(false)
{
}

//==============================================================================
// Counter Reading
//==============================================================================

RawCounters PerformanceCounters::read_raw() const {
    RawCounters raw{};
    
    if (axi_) {
        raw.total_cycles = axi_->read_reg(csr::PERF_TOTAL);
        raw.active_cycles = axi_->read_reg(csr::PERF_ACTIVE);
        raw.idle_cycles = axi_->read_reg(csr::PERF_IDLE);
        raw.cache_hits = axi_->read_reg(csr::PERF_CACHE_HITS);
        raw.cache_misses = axi_->read_reg(csr::PERF_CACHE_MISSES);
        raw.decode_count = axi_->read_reg(csr::PERF_DECODE_COUNT);
    }
    
    return raw;
}

void PerformanceCounters::reset() {
    // Performance counters auto-reset on START command
    // We just clear our baseline
    start_counters_ = {};
    timing_active_ = false;
}

//==============================================================================
// Timing Measurement
//==============================================================================

void PerformanceCounters::start_timing() {
    start_counters_ = read_raw();
    timing_active_ = true;
}

PerfMetrics PerformanceCounters::stop_timing(std::size_t data_bytes,
                                             std::uint64_t mac_ops) {
    RawCounters current = read_raw();
    RawCounters delta = current - start_counters_;
    timing_active_ = false;
    
    return compute_metrics(delta, data_bytes, mac_ops);
}

PerfMetrics PerformanceCounters::read_metrics(std::size_t data_bytes,
                                              std::uint64_t mac_ops) const {
    RawCounters current = read_raw();
    RawCounters delta = timing_active_ ? (current - start_counters_) : current;
    
    return compute_metrics(delta, data_bytes, mac_ops);
}

//==============================================================================
// Metric Computation
//==============================================================================

PerfMetrics PerformanceCounters::compute_metrics(const RawCounters& raw,
                                                 std::size_t data_bytes,
                                                 std::uint64_t mac_ops) const {
    PerfMetrics metrics{};
    
    // Store raw data
    metrics.data_bytes = data_bytes;
    
    // If MAC ops not provided, estimate from active cycles and PEs
    if (mac_ops == 0) {
        // Assume full PE utilization during active cycles
        mac_ops = static_cast<std::uint64_t>(raw.active_cycles) * num_pes_;
    }
    metrics.mac_operations = mac_ops;
    
    // Basic metrics
    if (raw.total_cycles > 0) {
        metrics.utilization = static_cast<double>(raw.active_cycles) / 
                             static_cast<double>(raw.total_cycles);
        metrics.stall_ratio = static_cast<double>(raw.idle_cycles) / 
                             static_cast<double>(raw.total_cycles);
    }
    
    // PE efficiency: actual MACs vs theoretical max during active time
    if (raw.active_cycles > 0 && num_pes_ > 0) {
        std::uint64_t theoretical_max = 
            static_cast<std::uint64_t>(raw.active_cycles) * num_pes_;
        metrics.pe_efficiency = static_cast<double>(mac_ops) / 
                               static_cast<double>(theoretical_max);
    }
    
    // Latency in milliseconds
    if (clock_mhz_ > 0) {
        metrics.latency_ms = static_cast<double>(raw.total_cycles) / 
                            (clock_mhz_ * 1000.0);
    }
    
    // Throughput in GOPS
    // Each MAC = 2 operations (multiply + add)
    if (metrics.latency_ms > 0) {
        double latency_sec = metrics.latency_ms / 1000.0;
        double total_ops = static_cast<double>(mac_ops) * 2.0;
        metrics.throughput_gops = (total_ops / latency_sec) / 1e9;
    }
    
    // Cache hit rate
    std::uint32_t total_accesses = raw.cache_hits + raw.cache_misses;
    if (total_accesses > 0) {
        metrics.cache_hit_rate = static_cast<double>(raw.cache_hits) / 
                                static_cast<double>(total_accesses);
    }
    
    // Bandwidth in GB/s
    if (metrics.latency_ms > 0 && data_bytes > 0) {
        double latency_sec = metrics.latency_ms / 1000.0;
        metrics.bandwidth_gbps = (static_cast<double>(data_bytes) / latency_sec) / 1e9;
    }
    
    // Operational intensity: FLOPS per byte
    if (data_bytes > 0) {
        double total_flops = static_cast<double>(mac_ops) * 2.0;
        metrics.operational_intensity = total_flops / static_cast<double>(data_bytes);
    }
    
    return metrics;
}

//==============================================================================
// Reporting
//==============================================================================

void PerformanceCounters::print_report(const PerfMetrics& metrics, 
                                       std::ostream& os) {
    os << "\n";
    os << "╔═══════════════════════════════════════════════════════════════╗\n";
    os << "║           PERFORMANCE REPORT                                  ║\n";
    os << "╠═══════════════════════════════════════════════════════════════╣\n";
    
    // Throughput and Latency
    os << "║ Throughput:        " << std::setw(10) << std::fixed << std::setprecision(3)
       << metrics.throughput_gops << " GOPS                        ║\n";
    os << "║ Latency:           " << std::setw(10) << std::fixed << std::setprecision(3)
       << metrics.latency_ms << " ms                          ║\n";
    
    // Utilization
    os << "╠═══════════════════════════════════════════════════════════════╣\n";
    os << "║ Utilization:       " << std::setw(10) << std::fixed << std::setprecision(1)
       << (metrics.utilization * 100.0) << " %                          ║\n";
    os << "║ PE Efficiency:     " << std::setw(10) << std::fixed << std::setprecision(1)
       << (metrics.pe_efficiency * 100.0) << " %                          ║\n";
    os << "║ Stall Ratio:       " << std::setw(10) << std::fixed << std::setprecision(1)
       << (metrics.stall_ratio * 100.0) << " %                          ║\n";
    
    // Cache
    if (metrics.cache_hit_rate > 0) {
        os << "╠═══════════════════════════════════════════════════════════════╣\n";
        os << "║ Cache Hit Rate:    " << std::setw(10) << std::fixed << std::setprecision(1)
           << (metrics.cache_hit_rate * 100.0) << " %                          ║\n";
    }
    
    // Memory
    if (metrics.data_bytes > 0) {
        os << "╠═══════════════════════════════════════════════════════════════╣\n";
        os << "║ Data Transferred:  " << std::setw(10) << (metrics.data_bytes / 1024)
           << " KB                          ║\n";
        os << "║ Bandwidth:         " << std::setw(10) << std::fixed << std::setprecision(2)
           << metrics.bandwidth_gbps << " GB/s                       ║\n";
        os << "║ Op Intensity:      " << std::setw(10) << std::fixed << std::setprecision(2)
           << metrics.operational_intensity << " FLOPS/byte              ║\n";
    }
    
    // Operations
    os << "╠═══════════════════════════════════════════════════════════════╣\n";
    os << "║ MAC Operations:    " << std::setw(10) << metrics.mac_operations
       << "                                ║\n";
    
    os << "╚═══════════════════════════════════════════════════════════════╝\n";
    os << std::endl;
}

std::string PerformanceCounters::summary(const PerfMetrics& metrics) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << metrics.throughput_gops << " GOPS, ";
    oss << metrics.latency_ms << " ms, ";
    oss << (metrics.utilization * 100.0) << "% util";
    return oss.str();
}

} // namespace resnet_accel
