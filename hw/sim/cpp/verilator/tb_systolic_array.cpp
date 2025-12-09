/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                      TB_SYSTOLIC_ARRAY.CPP                                ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  VERILATOR TESTBENCH: 16x16 Systolic Array unit test                     ║
 * ║  TESTS RTL MODULE: systolic_array.sv                                     ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║  Tests the systolic array in isolation to verify the row-stationary      ║
 * ║  dataflow, weight preloading, activation streaming, and partial sum      ║
 * ║  accumulation. This is the core compute unit of the accelerator.         ║
 * ║                                                                           ║
 * ║  REPLACES PYTHON: sw/tests/test_systolic.py                              ║
 * ║                                                                           ║
 * ║  WHY C++ INSTEAD OF PYTHON:                                              ║
 * ║  - Direct bit-level control over input timing                            ║
 * ║  - Can verify exact cycle-by-cycle behavior                              ║
 * ║  - Faster iteration for dataflow debugging                               ║
 * ║                                                                           ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  ARCHITECTURE NOTES:                                                      ║
 * ║                                                                           ║
 * ║  16x16 Systolic Array with Row-Stationary Dataflow:                      ║
 * ║                                                                           ║
 * ║     Activations flow →                                                    ║
 * ║    ┌────┬────┬────┬────┐                                                 ║
 * ║  W │PE00│PE01│PE02│... │ → psum                                          ║
 * ║  e ├────┼────┼────┼────┤                                                 ║
 * ║  i │PE10│PE11│PE12│... │ → psum                                          ║
 * ║  g ├────┼────┼────┼────┤                                                 ║
 * ║  h │... │... │... │... │ → psum                                          ║
 * ║  t └────┴────┴────┴────┘                                                 ║
 * ║  s                                                                        ║
 * ║  ↓                                                                        ║
 * ║                                                                           ║
 * ║  Each PE:                                                                 ║
 * ║  - Stores one weight (preloaded)                                         ║
 * ║  - Receives activation from left                                         ║
 * ║  - Passes activation to right                                            ║
 * ║  - Accumulates: psum += weight * activation                              ║
 * ║  - Outputs psum to accumulator                                           ║
 * ║                                                                           ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  WHAT YOU NEED TO IMPLEMENT:                                              ║
 * ║                                                                           ║
 * ║  1. Weight preload sequence                                               ║
 * ║     - Assert weight_load signal                                          ║
 * ║     - Stream 16x16 weights column by column                              ║
 * ║     - Deassert weight_load                                               ║
 * ║                                                                           ║
 * ║  2. Activation streaming                                                  ║
 * ║     - Stream activations row by row with skew                            ║
 * ║     - Row 0 starts at cycle 0                                            ║
 * ║     - Row 1 starts at cycle 1 (1 cycle delay)                            ║
 * ║     - Row N starts at cycle N                                            ║
 * ║                                                                           ║
 * ║  3. Result collection                                                     ║
 * ║     - Wait for pipeline to drain (16 + 16 - 1 = 31 cycles)              ║
 * ║     - Collect partial sums from each row                                 ║
 * ║     - Reshape to output matrix                                           ║
 * ║                                                                           ║
 * ║  4. Test cases                                                            ║
 * ║     - test_identity_weights(): Weight = I, output = input                ║
 * ║     - test_all_ones(): All 1s, output = N (row sum)                      ║
 * ║     - test_random(): Random matrices, compare to golden                  ║
 * ║     - test_overflow(): Near INT8_MAX values                              ║
 * ║                                                                           ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  KEY SIGNALS:                                                             ║
 * ║                                                                           ║
 * ║  Inputs:                                                                  ║
 * ║    clk, rst_n                                                            ║
 * ║    weight_load      - Assert during weight preload phase                 ║
 * ║    weight_data[127:0] - 16 x 8-bit weights per cycle                     ║
 * ║    act_valid        - Activation data valid                              ║
 * ║    act_data[127:0]  - 16 x 8-bit activations per cycle                   ║
 * ║                                                                           ║
 * ║  Outputs:                                                                 ║
 * ║    psum_valid       - Partial sum output valid                           ║
 * ║    psum_data[511:0] - 16 x 32-bit partial sums                           ║
 * ║    ready            - Ready to accept new data                           ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include <verilated.h>
#include <verilated_vcd_c.h>
// #include "Vsystolic_array.h"

#include <iostream>
#include <vector>
#include <random>
#include <cstring>

#include "../include/golden_models.hpp"
#include "../include/test_utils.hpp"

// =============================================================================
// Configuration
// =============================================================================

static constexpr int N = 16;  // Array dimension
static constexpr int PIPELINE_DEPTH = 2 * N - 1;  // 31 cycles

// =============================================================================
// Simulation State
// =============================================================================

// Vsystolic_array* dut = nullptr;
VerilatedVcdC* tfp = nullptr;
uint64_t sim_time = 0;

// =============================================================================
// Helpers
// =============================================================================

void tick() {
    // TODO: Implement clock toggle
}

void reset() {
    // TODO: Implement reset sequence
}

void preload_weights(const int8_t weights[N][N]) {
    // TODO: Implement weight preloading
    //
    // dut->weight_load = 1;
    //
    // // Load column by column (or row by row depending on RTL)
    // for (int col = 0; col < N; col++) {
    //     // Pack 16 weights into 128-bit bus
    //     __uint128_t packed = 0;
    //     for (int row = 0; row < N; row++) {
    //         packed |= ((__uint128_t)(uint8_t)weights[row][col]) << (row * 8);
    //     }
    //     dut->weight_data = packed;
    //     tick();
    // }
    //
    // dut->weight_load = 0;
}

void stream_activations(const int8_t acts[N][N], int32_t output[N][N]) {
    // TODO: Implement activation streaming with skew
    //
    // std::vector<std::vector<int32_t>> psum_buffer(N, std::vector<int32_t>(N, 0));
    // int output_col = 0;
    //
    // // Total cycles = N (for input) + pipeline depth
    // for (int cycle = 0; cycle < N + PIPELINE_DEPTH; cycle++) {
    //     // Drive activations with skew
    //     __uint128_t act_packed = 0;
    //     bool any_valid = false;
    //
    //     for (int row = 0; row < N; row++) {
    //         int act_idx = cycle - row;  // Skew: row N starts N cycles late
    //         if (act_idx >= 0 && act_idx < N) {
    //             act_packed |= ((__uint128_t)(uint8_t)acts[row][act_idx]) << (row * 8);
    //             any_valid = true;
    //         }
    //     }
    //
    //     dut->act_data = act_packed;
    //     dut->act_valid = any_valid;
    //     tick();
    //
    //     // Collect outputs
    //     if (dut->psum_valid) {
    //         for (int row = 0; row < N; row++) {
    //             output[row][output_col] = (dut->psum_data >> (row * 32)) & 0xFFFFFFFF;
    //         }
    //         output_col++;
    //     }
    // }
}

// =============================================================================
// Test Cases
// =============================================================================

bool test_identity_weights() {
    std::cout << "=== Test: Identity Weights ===" << std::endl;
    
    // TODO: Implement
    // int8_t weights[N][N] = {0};
    // for (int i = 0; i < N; i++) weights[i][i] = 1;  // Identity matrix
    //
    // int8_t acts[N][N];
    // for (int i = 0; i < N; i++)
    //     for (int j = 0; j < N; j++)
    //         acts[i][j] = i * N + j;
    //
    // preload_weights(weights);
    // int32_t output[N][N];
    // stream_activations(acts, output);
    //
    // // With identity weights, output should equal input
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         if (output[i][j] != acts[i][j]) {
    //             std::cerr << "FAIL at [" << i << "][" << j << "]" << std::endl;
    //             return false;
    //         }
    //     }
    // }
    
    std::cout << "PASS" << std::endl;
    return true;
}

bool test_all_ones() {
    std::cout << "=== Test: All Ones ===" << std::endl;
    
    // TODO: Implement
    // All weights = 1, all activations = 1
    // Output should be N for each element (sum of N ones)
    
    std::cout << "PASS" << std::endl;
    return true;
}

bool test_random_matrices() {
    std::cout << "=== Test: Random Matrices ===" << std::endl;
    
    // TODO: Implement
    // std::mt19937 rng(42);
    // std::uniform_int_distribution<int> dist(-128, 127);
    //
    // int8_t weights[N][N], acts[N][N];
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         weights[i][j] = dist(rng);
    //         acts[i][j] = dist(rng);
    //     }
    // }
    //
    // // Golden model
    // int32_t golden[N][N];
    // golden::matmul_int8(&acts[0][0], &weights[0][0], &golden[0][0], N, N, N);
    //
    // // Hardware
    // preload_weights(weights);
    // int32_t output[N][N];
    // stream_activations(acts, output);
    //
    // // Compare
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         if (output[i][j] != golden[i][j]) {
    //             std::cerr << "FAIL at [" << i << "][" << j << "]" << std::endl;
    //             return false;
    //         }
    //     }
    // }
    
    std::cout << "PASS" << std::endl;
    return true;
}

bool test_overflow_handling() {
    std::cout << "=== Test: Overflow Handling ===" << std::endl;
    
    // TODO: Implement
    // Use maximum values to verify INT32 accumulator doesn't overflow
    // 127 * 127 * 16 = 258064, which fits in INT32
    
    std::cout << "PASS" << std::endl;
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    // TODO: Implement
    // dut = new Vsystolic_array;
    // Verilated::traceEverOn(true);
    // tfp = new VerilatedVcdC;
    // dut->trace(tfp, 99);
    // tfp->open("systolic_array.vcd");
    //
    // reset();
    //
    // bool pass = true;
    // pass &= test_identity_weights();
    // pass &= test_all_ones();
    // pass &= test_random_matrices();
    // pass &= test_overflow_handling();
    //
    // tfp->close();
    // delete tfp;
    // delete dut;
    //
    // return pass ? 0 : 1;
    
    std::cout << "tb_systolic_array: Not yet implemented" << std::endl;
    return 0;
}
