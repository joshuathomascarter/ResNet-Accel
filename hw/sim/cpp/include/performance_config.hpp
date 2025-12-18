/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                       PERFORMANCE_CONFIG.HPP                              ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  Clock and performance targets for different FPGA platforms              ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  SUPPORTED PLATFORMS:                                                     ║
 * ║    - PYNQ-Z2 (Zynq-7020): 100-150 MHz achievable                         ║
 * ║    - ZCU104 (UltraScale+): 200 MHz achievable                            ║
 * ║    - ZCU102 (UltraScale+): 250 MHz achievable                            ║
 * ║    - Alveo U50: 300 MHz achievable                                       ║
 * ║                                                                           ║
 * ║  PEAK THROUGHPUT FORMULA:                                                 ║
 * ║    GOPS = Array_Size² × 2 × Clock_MHz / 1000                             ║
 * ║                                                                           ║
 * ║  For 14×14 array:                                                         ║
 * ║    @ 100 MHz: 196 × 2 × 100 = 39.2 GOPS                                  ║
 * ║    @ 150 MHz: 196 × 2 × 150 = 58.8 GOPS                                  ║
 * ║    @ 200 MHz: 196 × 2 × 200 = 78.4 GOPS                                  ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#ifndef PERFORMANCE_CONFIG_HPP
#define PERFORMANCE_CONFIG_HPP

#include <cstdint>
#include <string>

namespace resnet_accel {

// =============================================================================
// Systolic Array Configuration
// =============================================================================

/// Array dimensions (14×14 for Zynq-7020 DSP count)
constexpr int ARRAY_ROWS = 14;
constexpr int ARRAY_COLS = 14;
constexpr int ARRAY_SIZE = ARRAY_ROWS;  // Alias for square arrays
constexpr int NUM_PES = ARRAY_ROWS * ARRAY_COLS;  // 196 PEs

/// Operations per PE per cycle (1 multiply + 1 accumulate = 2 ops)
constexpr int OPS_PER_PE = 2;

/// Total operations per cycle for the entire array
constexpr int OPS_PER_CYCLE = NUM_PES * OPS_PER_PE;  // 392 ops/cycle

// =============================================================================
// Platform Configuration Structure
// =============================================================================

struct PlatformConfig {
    const char* name;           ///< Human-readable platform name
    const char* device;         ///< Xilinx device part number
    double clock_mhz;           ///< Target clock frequency
    double peak_gops;           ///< Peak throughput in GOPS
    double power_watts;         ///< Estimated power consumption
    double gops_per_watt;       ///< Energy efficiency
    bool timing_achievable;     ///< Whether timing closure is likely
    const char* notes;          ///< Additional notes
};

// =============================================================================
// FPGA Platform Configurations
// =============================================================================

namespace platform {

// ─────────────────────────────────────────────────────────────────────────────
// PYNQ-Z2 (Zynq-7020, XC7Z020-1CLG400C, -1 speed grade)
// ─────────────────────────────────────────────────────────────────────────────

/// Conservative: Guaranteed timing closure
constexpr PlatformConfig PYNQ_Z2_CONSERVATIVE = {
    .name = "PYNQ-Z2 (Conservative)",
    .device = "XC7Z020-1CLG400C",
    .clock_mhz = 100.0,
    .peak_gops = 39.2,   // 196 × 2 × 100 / 1000
    .power_watts = 2.0,
    .gops_per_watt = 19.6,
    .timing_achievable = true,
    .notes = "Safe default, always meets timing"
};

/// Optimized: With careful RTL and constraints
constexpr PlatformConfig PYNQ_Z2_OPTIMIZED = {
    .name = "PYNQ-Z2 (Optimized)",
    .device = "XC7Z020-1CLG400C",
    .clock_mhz = 150.0,
    .peak_gops = 58.8,   // 196 × 2 × 150 / 1000
    .power_watts = 2.5,
    .gops_per_watt = 23.5,
    .timing_achievable = true,
    .notes = "Requires timing optimization, achievable with effort"
};

/// Aggressive: May require timing waivers
constexpr PlatformConfig PYNQ_Z2_AGGRESSIVE = {
    .name = "PYNQ-Z2 (Aggressive)",
    .device = "XC7Z020-1CLG400C",
    .clock_mhz = 175.0,
    .peak_gops = 68.6,   // 196 × 2 × 175 / 1000
    .power_watts = 3.0,
    .gops_per_watt = 22.9,
    .timing_achievable = false,
    .notes = "Risky - may have timing violations"
};

// ─────────────────────────────────────────────────────────────────────────────
// ZCU104 (Zynq UltraScale+ XCZU7EV)
// ─────────────────────────────────────────────────────────────────────────────

constexpr PlatformConfig ZCU104 = {
    .name = "ZCU104 (UltraScale+)",
    .device = "XCZU7EV-2FFVC1156",
    .clock_mhz = 200.0,
    .peak_gops = 78.4,   // 196 × 2 × 200 / 1000
    .power_watts = 5.0,
    .gops_per_watt = 15.7,
    .timing_achievable = true,
    .notes = "Production target, achievable with UltraScale+"
};

// ─────────────────────────────────────────────────────────────────────────────
// ZCU102 (Zynq UltraScale+ XCZU9EG)
// ─────────────────────────────────────────────────────────────────────────────

constexpr PlatformConfig ZCU102 = {
    .name = "ZCU102 (UltraScale+)",
    .device = "XCZU9EG-2FFVB1156",
    .clock_mhz = 250.0,
    .peak_gops = 98.0,   // 196 × 2 × 250 / 1000
    .power_watts = 8.0,
    .gops_per_watt = 12.25,
    .timing_achievable = true,
    .notes = "High-end development board"
};

// ─────────────────────────────────────────────────────────────────────────────
// Alveo U50 (Data Center Accelerator)
// ─────────────────────────────────────────────────────────────────────────────

constexpr PlatformConfig ALVEO_U50 = {
    .name = "Alveo U50",
    .device = "XCU50-2FSVH2104",
    .clock_mhz = 300.0,
    .peak_gops = 117.6,  // 196 × 2 × 300 / 1000
    .power_watts = 25.0,
    .gops_per_watt = 4.7,
    .timing_achievable = true,
    .notes = "Data center grade, HBM memory"
};

}  // namespace platform

// =============================================================================
// Default Platform Selection
// =============================================================================

// Select default based on compile-time target
#if defined(TARGET_ZCU104)
    constexpr auto DEFAULT_PLATFORM = platform::ZCU104;
    constexpr double DEFAULT_CLOCK_MHZ = 200.0;
#elif defined(TARGET_ZCU102)
    constexpr auto DEFAULT_PLATFORM = platform::ZCU102;
    constexpr double DEFAULT_CLOCK_MHZ = 250.0;
#elif defined(TARGET_ALVEO_U50)
    constexpr auto DEFAULT_PLATFORM = platform::ALVEO_U50;
    constexpr double DEFAULT_CLOCK_MHZ = 300.0;
#elif defined(TARGET_PYNQ_Z2_OPTIMIZED)
    constexpr auto DEFAULT_PLATFORM = platform::PYNQ_Z2_OPTIMIZED;
    constexpr double DEFAULT_CLOCK_MHZ = 150.0;
#else
    // Default: PYNQ-Z2 Conservative (most common student board)
    constexpr auto DEFAULT_PLATFORM = platform::PYNQ_Z2_CONSERVATIVE;
    constexpr double DEFAULT_CLOCK_MHZ = 100.0;
#endif

// =============================================================================
// Performance Calculation Functions
// =============================================================================

/// Calculate peak GOPS for a given clock frequency
constexpr double calculate_peak_gops(double clock_mhz) {
    return static_cast<double>(NUM_PES) * OPS_PER_PE * clock_mhz / 1000.0;
}

/// Calculate theoretical minimum cycles for a GEMM operation
/// For C[M×N] = A[M×K] × B[K×N]
constexpr uint64_t calculate_min_cycles(uint32_t M, uint32_t N, uint32_t K) {
    // Each output requires K MACs
    // Array can compute ARRAY_ROWS × ARRAY_COLS outputs per cycle (with perfect reuse)
    // Tiled execution: ceil(M/Tm) × ceil(N/Tn) × ceil(K/Tk) tiles
    uint64_t total_macs = static_cast<uint64_t>(M) * N * K;
    return total_macs / NUM_PES;  // Ideal minimum (100% utilization)
}

/// Calculate inference time in milliseconds
constexpr double calculate_inference_time_ms(double clock_mhz, uint64_t total_ops, 
                                              double utilization = 0.80) {
    double gops = calculate_peak_gops(clock_mhz) * utilization;
    return static_cast<double>(total_ops) / (gops * 1e6);
}

// =============================================================================
// ResNet-18 Operation Counts
// =============================================================================

namespace resnet18 {

/// Approximate total MACs for ResNet-18 inference (single image)
constexpr uint64_t TOTAL_MACS = 1800000000ULL;  // ~1.8 billion MACs

/// Total operations (MAC = 1 multiply + 1 add)
constexpr uint64_t TOTAL_OPS = TOTAL_MACS * 2;  // ~3.6 billion ops

/// Calculate ResNet-18 inference time in milliseconds
constexpr double inference_time_ms(double clock_mhz, double utilization = 0.80) {
    double peak_gops = calculate_peak_gops(clock_mhz);
    double effective_gops = peak_gops * utilization;
    return static_cast<double>(TOTAL_OPS) / (effective_gops * 1e6);
}

/// Calculate frames per second for ResNet-18
constexpr double inference_fps(double clock_mhz, double utilization = 0.80) {
    return 1000.0 / inference_time_ms(clock_mhz, utilization);
}

/// Layer-by-layer breakdown (approximate)
namespace layers {
    constexpr uint64_t CONV1_MACS   = 118013952ULL;   // 7×7 conv, 64 filters
    constexpr uint64_t LAYER1_MACS  = 231211008ULL;   // 2× BasicBlock
    constexpr uint64_t LAYER2_MACS  = 462422016ULL;   // 2× BasicBlock
    constexpr uint64_t LAYER3_MACS  = 462422016ULL;   // 2× BasicBlock
    constexpr uint64_t LAYER4_MACS  = 462422016ULL;   // 2× BasicBlock
    constexpr uint64_t FC_MACS      = 512000ULL;      // 512 → 1000 FC
}

}  // namespace resnet18

// =============================================================================
// MNIST CNN Operation Counts (for testing)
// =============================================================================

namespace mnist_cnn {

/// Simple 4-layer CNN for MNIST
constexpr uint64_t CONV1_MACS = 1 * 32 * 26 * 26 * 3 * 3;      // ~219K
constexpr uint64_t CONV2_MACS = 32 * 64 * 11 * 11 * 3 * 3;     // ~2.2M
constexpr uint64_t FC1_MACS = 64 * 5 * 5 * 128;                 // ~205K
constexpr uint64_t FC2_MACS = 128 * 10;                         // ~1.3K

constexpr uint64_t TOTAL_MACS = CONV1_MACS + CONV2_MACS + FC1_MACS + FC2_MACS;
constexpr uint64_t TOTAL_OPS = TOTAL_MACS * 2;

constexpr double inference_time_ms(double clock_mhz, double utilization = 0.80) {
    double peak_gops = calculate_peak_gops(clock_mhz);
    double effective_gops = peak_gops * utilization;
    return static_cast<double>(TOTAL_OPS) / (effective_gops * 1e6);
}

}  // namespace mnist_cnn

// =============================================================================
// Performance Summary Table (for printing)
// =============================================================================

inline void print_platform_table() {
    const PlatformConfig platforms[] = {
        platform::PYNQ_Z2_CONSERVATIVE,
        platform::PYNQ_Z2_OPTIMIZED,
        platform::PYNQ_Z2_AGGRESSIVE,
        platform::ZCU104,
        platform::ZCU102,
        platform::ALVEO_U50
    };
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    ACCELERATOR PERFORMANCE BY PLATFORM                        ║\n");
    printf("║                    Array: %d×%d = %d PEs                                       ║\n", 
           ARRAY_ROWS, ARRAY_COLS, NUM_PES);
    printf("╠═══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Platform                  │ Clock   │ Peak GOPS │ Power  │ Eff.    │ Timing  ║\n");
    printf("╠───────────────────────────┼─────────┼───────────┼────────┼─────────┼─────────╣\n");
    
    for (const auto& p : platforms) {
        printf("║ %-25s │ %3.0f MHz │ %5.1f     │ %4.1f W │ %4.1f    │ %s     ║\n",
               p.name, p.clock_mhz, p.peak_gops, p.power_watts, p.gops_per_watt,
               p.timing_achievable ? "✅" : "⚠️ ");
    }
    
    printf("╚═══════════════════════════════════════════════════════════════════════════════╝\n");
}

inline void print_resnet18_estimates() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    RESNET-18 INFERENCE ESTIMATES                              ║\n");
    printf("║                    (Assuming 80%% utilization)                                 ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Total Operations: %llu (~3.6 billion)                           ║\n", 
           (unsigned long long)resnet18::TOTAL_OPS);
    printf("╠───────────────────────────┬────────────┬──────────┬────────────────────────────╣\n");
    printf("║ Platform                  │ Time       │ FPS      │ Real-time?                 ║\n");
    printf("╠───────────────────────────┼────────────┼──────────┼────────────────────────────╣\n");
    
    auto print_row = [](const PlatformConfig& p) {
        double time_ms = resnet18::inference_time_ms(p.clock_mhz, 0.80);
        double fps = resnet18::inference_fps(p.clock_mhz, 0.80);
        const char* status = fps >= 30 ? "✅ Real-time (30+ FPS)" : 
                            (fps >= 10 ? "⚡ Interactive" : "📊 Batch");
        printf("║ %-25s │ %6.1f ms  │ %5.1f    │ %-26s ║\n",
               p.name, time_ms, fps, status);
    };
    
    print_row(platform::PYNQ_Z2_CONSERVATIVE);
    print_row(platform::PYNQ_Z2_OPTIMIZED);
    print_row(platform::ZCU104);
    print_row(platform::ZCU102);
    print_row(platform::ALVEO_U50);
    
    printf("╚═══════════════════════════════════════════════════════════════════════════════╝\n");
}

}  // namespace resnet_accel

#endif // PERFORMANCE_CONFIG_HPP
