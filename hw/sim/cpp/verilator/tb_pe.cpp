/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                            TB_PE.CPP                                      ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  VERILATOR TESTBENCH: Single Processing Element unit test                ║
 * ║  TESTS RTL MODULE: pe.sv                                                 ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║  Tests a single PE (Processing Element) which is the basic building     ║
 * ║  block of the systolic array. Each PE stores a weight, receives         ║
 * ║  activations, computes MAC, and passes data to neighbors.               ║
 * ║                                                                           ║
 * ║  REPLACES PYTHON: sw/tests/test_pe.py                                    ║
 * ║                                                                           ║
 * ║  WHY C++ INSTEAD OF PYTHON:                                              ║
 * ║  - Verify exact timing of data propagation                               ║
 * ║  - Test weight stationary behavior                                        ║
 * ║  - Debug activation forwarding logic                                      ║
 * ║                                                                           ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  PE ARCHITECTURE:                                                         ║
 * ║                                                                           ║
 * ║                    weight_in                                              ║
 * ║                        │                                                  ║
 * ║                        ▼                                                  ║
 * ║                 ┌──────────────┐                                          ║
 * ║   act_in ────►  │   Weight     │                                          ║
 * ║                 │   Register   │                                          ║
 * ║                 └──────┬───────┘                                          ║
 * ║                        │                                                  ║
 * ║                        ▼                                                  ║
 * ║                 ┌──────────────┐                                          ║
 * ║                 │   MAC Unit   │ ◄──── psum_in (from PE above)           ║
 * ║                 └──────┬───────┘                                          ║
 * ║                        │                                                  ║
 * ║                        ▼                                                  ║
 * ║                    psum_out ────► (to PE below or output)                ║
 * ║                                                                           ║
 * ║   act_out ────► (to PE on right, delayed by 1 cycle)                     ║
 * ║                                                                           ║
 * ║  Row-Stationary Dataflow:                                                 ║
 * ║  - Weight is loaded once and stays                                       ║
 * ║  - Activations stream through horizontally                               ║
 * ║  - Partial sums accumulate vertically                                    ║
 * ║                                                                           ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  WHAT YOU NEED TO IMPLEMENT:                                              ║
 * ║                                                                           ║
 * ║  1. Weight loading test                                                   ║
 * ║     - Load weight, verify it's stored                                    ║
 * ║     - Verify weight persists across cycles                               ║
 * ║                                                                           ║
 * ║  2. Activation forwarding test                                            ║
 * ║     - Send activation in, verify it appears at act_out next cycle        ║
 * ║     - Verify multiple activations pipeline correctly                     ║
 * ║                                                                           ║
 * ║  3. MAC computation test                                                  ║
 * ║     - Load weight, send activation                                       ║
 * ║     - Verify psum_out = psum_in + weight * activation                    ║
 * ║                                                                           ║
 * ║  4. Full pipeline test                                                    ║
 * ║     - Stream of activations                                              ║
 * ║     - Verify each output                                                 ║
 * ║                                                                           ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  KEY SIGNALS:                                                             ║
 * ║                                                                           ║
 * ║  Inputs:                                                                  ║
 * ║    clk, rst_n                                                            ║
 * ║    weight_load      - Load new weight                                    ║
 * ║    weight_in[7:0]   - Weight value to load                               ║
 * ║    act_in[7:0]      - Activation from left PE                            ║
 * ║    act_valid_in     - Activation valid                                   ║
 * ║    psum_in[31:0]    - Partial sum from above PE                          ║
 * ║                                                                           ║
 * ║  Outputs:                                                                 ║
 * ║    act_out[7:0]     - Activation to right PE (1 cycle delay)             ║
 * ║    act_valid_out    - Activation valid out                               ║
 * ║    psum_out[31:0]   - Partial sum to below PE                            ║
 * ║    psum_valid_out   - Partial sum valid                                  ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include <verilated.h>
#include <verilated_vcd_c.h>
// #include "Vpe.h"

#include <iostream>
#include <cstdint>
#include <queue>

// =============================================================================
// Simulation State
// =============================================================================

// Vpe* dut = nullptr;
uint64_t sim_time = 0;

// =============================================================================
// Helpers
// =============================================================================

void tick() {
    // TODO: Implement clock toggle
}

void reset() {
    // TODO: Implement reset
}

void load_weight(int8_t weight) {
    // TODO: Implement
    // dut->weight_load = 1;
    // dut->weight_in = weight;
    // tick();
    // dut->weight_load = 0;
}

void send_activation(int8_t act, int32_t psum_in = 0) {
    // TODO: Implement
    // dut->act_in = act;
    // dut->act_valid_in = 1;
    // dut->psum_in = psum_in;
    // tick();
    // dut->act_valid_in = 0;
}

// =============================================================================
// Test Cases
// =============================================================================

bool test_weight_load() {
    std::cout << "=== Test: Weight Load ===" << std::endl;
    
    // TODO: Implement
    // load_weight(42);
    //
    // // Send a zero activation to read out the weight effect
    // send_activation(1, 0);
    //
    // // psum_out should be 42 (42 * 1 + 0)
    // if (dut->psum_out != 42) {
    //     std::cerr << "FAIL: Weight not loaded correctly" << std::endl;
    //     return false;
    // }
    
    std::cout << "PASS" << std::endl;
    return true;
}

bool test_weight_persistence() {
    std::cout << "=== Test: Weight Persistence ===" << std::endl;
    
    // TODO: Implement
    // load_weight(10);
    //
    // // Send multiple activations, weight should stay
    // for (int i = 0; i < 5; i++) {
    //     send_activation(i + 1, 0);
    //     if (dut->psum_out != 10 * (i + 1)) {
    //         std::cerr << "FAIL: Weight changed unexpectedly" << std::endl;
    //         return false;
    //     }
    // }
    
    std::cout << "PASS" << std::endl;
    return true;
}

bool test_activation_forwarding() {
    std::cout << "=== Test: Activation Forwarding ===" << std::endl;
    
    // TODO: Implement
    // load_weight(0);  // Zero weight to isolate forwarding
    //
    // std::queue<int8_t> expected;
    //
    // for (int8_t i = 1; i <= 10; i++) {
    //     send_activation(i, 0);
    //
    //     // Check if previous activation appears at output
    //     if (!expected.empty()) {
    //         if (dut->act_out != expected.front()) {
    //             std::cerr << "FAIL: Activation forwarding error" << std::endl;
    //             return false;
    //         }
    //         expected.pop();
    //     }
    //     expected.push(i);
    // }
    
    std::cout << "PASS" << std::endl;
    return true;
}

bool test_mac_computation() {
    std::cout << "=== Test: MAC Computation ===" << std::endl;
    
    // TODO: Implement
    // load_weight(7);
    //
    // // Test: psum_out = psum_in + weight * activation
    // // = 100 + 7 * 5 = 135
    // send_activation(5, 100);
    //
    // if (dut->psum_out != 135) {
    //     std::cerr << "FAIL: MAC computation error" << std::endl;
    //     return false;
    // }
    
    std::cout << "PASS" << std::endl;
    return true;
}

bool test_negative_values() {
    std::cout << "=== Test: Negative Values ===" << std::endl;
    
    // TODO: Implement
    // load_weight(-10);
    // send_activation(-5, 50);
    // // Expected: 50 + (-10) * (-5) = 50 + 50 = 100
    //
    // if (dut->psum_out != 100) {
    //     std::cerr << "FAIL: Negative value handling error" << std::endl;
    //     return false;
    // }
    
    std::cout << "PASS" << std::endl;
    return true;
}

bool test_psum_accumulation() {
    std::cout << "=== Test: Partial Sum Accumulation ===" << std::endl;
    
    // TODO: Implement
    // load_weight(1);
    //
    // int32_t running_psum = 0;
    // for (int8_t i = 1; i <= 10; i++) {
    //     send_activation(i, running_psum);
    //     running_psum = dut->psum_out;
    // }
    //
    // // Sum of 1 + 2 + ... + 10 = 55
    // if (running_psum != 55) {
    //     std::cerr << "FAIL: Accumulation error" << std::endl;
    //     return false;
    // }
    
    std::cout << "PASS" << std::endl;
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    // TODO: Implement
    // dut = new Vpe;
    // reset();
    //
    // bool pass = true;
    // pass &= test_weight_load();
    // pass &= test_weight_persistence();
    // pass &= test_activation_forwarding();
    // pass &= test_mac_computation();
    // pass &= test_negative_values();
    // pass &= test_psum_accumulation();
    //
    // delete dut;
    // return pass ? 0 : 1;
    
    std::cout << "tb_pe: Not yet implemented" << std::endl;
    return 0;
}
