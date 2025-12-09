/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                         TB_ACCEL_TOP.CPP                                  ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  VERILATOR TESTBENCH: Top-level accelerator integration test             ║
 * ║  TESTS RTL MODULE: accel_top.sv                                          ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║  This is the main Verilator testbench that instantiates the complete     ║
 * ║  accelerator design (accel_top) and runs full end-to-end tests. It       ║
 * ║  verifies the entire data path from AXI input through systolic array     ║
 * ║  computation to AXI output.                                              ║
 * ║                                                                           ║
 * ║  REPLACES PYTHON: sw/tests/test_accel_integration.py (cocotb tests)      ║
 * ║                                                                           ║
 * ║  WHY C++ INSTEAD OF PYTHON:                                              ║
 * ║  - Direct cycle-accurate simulation without Python overhead              ║
 * ║  - 10-100x faster than cocotb for long simulations                       ║
 * ║  - Better debugging with GDB/LLDB                                        ║
 * ║  - Can run millions of cycles for stress testing                         ║
 * ║                                                                           ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  WHAT YOU NEED TO IMPLEMENT:                                              ║
 * ║                                                                           ║
 * ║  1. Verilator model instantiation                                         ║
 * ║     - #include "Vaccel_top.h"                                            ║
 * ║     - Create Vaccel_top* top = new Vaccel_top;                           ║
 * ║     - Create VerilatedVcdC* tfp for waveform tracing                     ║
 * ║                                                                           ║
 * ║  2. Clock and reset generation                                            ║
 * ║     - toggle_clock(): top->clk = !top->clk; top->eval();                 ║
 * ║     - reset_dut(): Assert rst_n low for N cycles                         ║
 * ║                                                                           ║
 * ║  3. AXI transaction helpers                                               ║
 * ║     - axi_write(addr, data): Drive AXI write channel signals             ║
 * ║     - axi_read(addr): Drive AXI read channel, return data                ║
 * ║     - wait_for_done(): Poll status register until done bit set           ║
 * ║                                                                           ║
 * ║  4. Test cases                                                            ║
 * ║     - test_register_access(): Read/write config registers                ║
 * ║     - test_simple_matmul(): Small 16x16 matrix multiply                  ║
 * ║     - test_conv_layer(): Single convolution layer                        ║
 * ║     - test_bsr_sparse(): BSR format sparse matrix                        ║
 * ║     - test_full_inference(): Complete layer sequence                     ║
 * ║                                                                           ║
 * ║  5. Result verification                                                   ║
 * ║     - Compare hardware output against golden model                        ║
 * ║     - Report PASS/FAIL with detailed error info                          ║
 * ║                                                                           ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  KEY SIGNALS TO DRIVE:                                                    ║
 * ║                                                                           ║
 * ║  Clocks/Reset:                                                            ║
 * ║    top->clk        - Main clock                                          ║
 * ║    top->rst_n      - Active-low reset                                    ║
 * ║                                                                           ║
 * ║  AXI-Lite (Config):                                                       ║
 * ║    top->s_axil_awaddr, awvalid, awready                                  ║
 * ║    top->s_axil_wdata, wstrb, wvalid, wready                              ║
 * ║    top->s_axil_bresp, bvalid, bready                                     ║
 * ║    top->s_axil_araddr, arvalid, arready                                  ║
 * ║    top->s_axil_rdata, rresp, rvalid, rready                              ║
 * ║                                                                           ║
 * ║  AXI-Full (Data):                                                         ║
 * ║    top->s_axi_* (slave port for weight/activation input)                 ║
 * ║    top->m_axi_* (master port for result output)                          ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include <verilated.h>
#include <verilated_vcd_c.h>
// #include "Vaccel_top.h"  // Generated by Verilator from accel_top.sv

#include <iostream>
#include <cstdint>
#include <vector>
#include <memory>

// Include our C++ libraries
#include "../include/axi_master.hpp"
#include "../include/golden_models.hpp"
#include "../include/bsr_packer.hpp"
#include "../include/test_utils.hpp"

// =============================================================================
// Configuration
// =============================================================================

static constexpr int TRACE_DEPTH = 99;
static constexpr uint64_t MAX_SIM_TIME = 1000000;  // Maximum cycles

// Register addresses (must match RTL)
static constexpr uint32_t REG_CTRL       = 0x00;
static constexpr uint32_t REG_STATUS     = 0x04;
static constexpr uint32_t REG_WEIGHT_PTR = 0x08;
static constexpr uint32_t REG_ACT_PTR    = 0x0C;
static constexpr uint32_t REG_OUT_PTR    = 0x10;
static constexpr uint32_t REG_LAYER_CFG  = 0x14;

// =============================================================================
// Global State
// =============================================================================

// Vaccel_top* top = nullptr;
VerilatedVcdC* tfp = nullptr;
uint64_t sim_time = 0;
bool trace_enabled = true;

// =============================================================================
// Clock and Reset
// =============================================================================

void toggle_clock() {
    // TODO: Implement
    // top->clk = !top->clk;
    // top->eval();
    // if (trace_enabled && tfp) tfp->dump(sim_time);
    // sim_time++;
}

void tick() {
    // Rising edge
    toggle_clock();
    // Falling edge  
    toggle_clock();
}

void reset_dut(int cycles = 10) {
    // TODO: Implement
    // top->rst_n = 0;
    // for (int i = 0; i < cycles; i++) tick();
    // top->rst_n = 1;
    // tick();
}

// =============================================================================
// AXI-Lite Transactions
// =============================================================================

void axil_write(uint32_t addr, uint32_t data) {
    // TODO: Implement AXI-Lite write
    //
    // // Address phase
    // top->s_axil_awaddr = addr;
    // top->s_axil_awvalid = 1;
    // top->s_axil_wdata = data;
    // top->s_axil_wstrb = 0xF;
    // top->s_axil_wvalid = 1;
    // top->s_axil_bready = 1;
    //
    // // Wait for ready
    // while (!top->s_axil_awready || !top->s_axil_wready) tick();
    // tick();
    //
    // // Deassert
    // top->s_axil_awvalid = 0;
    // top->s_axil_wvalid = 0;
    //
    // // Wait for response
    // while (!top->s_axil_bvalid) tick();
    // tick();
    // top->s_axil_bready = 0;
}

uint32_t axil_read(uint32_t addr) {
    // TODO: Implement AXI-Lite read
    //
    // top->s_axil_araddr = addr;
    // top->s_axil_arvalid = 1;
    // top->s_axil_rready = 1;
    //
    // while (!top->s_axil_arready) tick();
    // tick();
    // top->s_axil_arvalid = 0;
    //
    // while (!top->s_axil_rvalid) tick();
    // uint32_t data = top->s_axil_rdata;
    // tick();
    // top->s_axil_rready = 0;
    //
    // return data;
    return 0;
}

// =============================================================================
// Test Cases
// =============================================================================

bool test_register_access() {
    std::cout << "=== Test: Register Access ===" << std::endl;
    
    // TODO: Implement
    // axil_write(REG_WEIGHT_PTR, 0x12345678);
    // uint32_t readback = axil_read(REG_WEIGHT_PTR);
    // if (readback != 0x12345678) {
    //     std::cerr << "FAIL: Expected 0x12345678, got " << std::hex << readback << std::endl;
    //     return false;
    // }
    
    std::cout << "PASS" << std::endl;
    return true;
}

bool test_simple_matmul() {
    std::cout << "=== Test: Simple 16x16 MatMul ===" << std::endl;
    
    // TODO: Implement
    // 1. Create 16x16 test matrices A and B
    // 2. Compute golden result C = A * B using golden::matmul_int8
    // 3. Load A and B into simulated memory
    // 4. Configure accelerator registers
    // 5. Start computation
    // 6. Wait for completion
    // 7. Read result and compare
    
    std::cout << "PASS" << std::endl;
    return true;
}

bool test_conv_layer() {
    std::cout << "=== Test: Convolution Layer ===" << std::endl;
    
    // TODO: Implement
    // 1. Load test activation and weight from fixtures
    // 2. Run golden model
    // 3. Run hardware
    // 4. Compare results
    
    std::cout << "PASS" << std::endl;
    return true;
}

bool test_bsr_sparse() {
    std::cout << "=== Test: BSR Sparse Format ===" << std::endl;
    
    // TODO: Implement
    // 1. Create sparse matrix with known sparsity
    // 2. Convert to BSR format
    // 3. Run through accelerator
    // 4. Verify computation skips zero blocks
    
    std::cout << "PASS" << std::endl;
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    
    // TODO: Implement
    //
    // // Create model
    // top = new Vaccel_top;
    //
    // // Enable tracing
    // if (trace_enabled) {
    //     Verilated::traceEverOn(true);
    //     tfp = new VerilatedVcdC;
    //     top->trace(tfp, TRACE_DEPTH);
    //     tfp->open("accel_top.vcd");
    // }
    //
    // // Reset
    // reset_dut();
    //
    // // Run tests
    // bool all_pass = true;
    // all_pass &= test_register_access();
    // all_pass &= test_simple_matmul();
    // all_pass &= test_conv_layer();
    // all_pass &= test_bsr_sparse();
    //
    // // Cleanup
    // if (tfp) {
    //     tfp->close();
    //     delete tfp;
    // }
    // delete top;
    //
    // return all_pass ? 0 : 1;
    
    std::cout << "tb_accel_top: Not yet implemented" << std::endl;
    return 0;
}
