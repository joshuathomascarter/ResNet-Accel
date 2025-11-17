#!/usr/bin/env python3
"""
AXI Master Integration Setup & Testing Guide
============================================================================

This document explains how to:
1. Set up Cocotb for Python ↔ Verilog integration testing
2. Run the enhanced AXI testbench
3. Verify your AXI implementation

Author: ACCEL-v1 Team
Date: 2025-11-16
"""

# ============================================================================
# SECTION 1: QUICK START
# ============================================================================

"""
Quick Start (TL;DR):

1. Install Cocotb:
   $ pip install cocotb

2. Run Cocotb tests:
   $ cd /workspaces/ACCEL-v1/accel\ v1
   $ make -f tb/Makefile.cocotb SIM=iverilog

3. View results:
   Check for "ALL PASSED" in console output

For more options, see the full guide below.
"""

# ============================================================================
# SECTION 2: PREREQUISITES & INSTALLATION
# ============================================================================

"""
Prerequisites:
- Python 3.7+
- Cocotb framework
- Verilog simulator (iverilog, Verilator, or other)

Installation Steps:

1. Install Cocotb:
   $ pip install cocotb==1.8.0

2. Verify installation:
   $ cocotb-config --version
   
   Expected output: cocotb-1.8.0

3. Ensure simulators are available:
   
   For iverilog:
   $ apt-get install iverilog gtkwave
   $ which iverilog
   
   For Verilator (optional, for better performance):
   $ apt-get install verilator
   $ which verilator
"""

# ============================================================================
# SECTION 3: DIRECTORY STRUCTURE
# ============================================================================

"""
Project structure for testing:

/workspaces/ACCEL-v1/accel v1/
├── tb/
│   ├── Makefile.cocotb              ← Cocotb configuration
│   ├── cocotb_axi_master_test.py    ← Python testbench (THIS FILE)
│   ├── unit/
│   │   └── ... other unit tests
│   └── integration/
│       └── ... integration tests
│
├── verilog/
│   ├── host_iface/
│   │   ├── axi_lite_slave.sv         ← DUT
│   │   ├── tb_axi_lite_slave.sv      ← Original testbench
│   │   └── tb_axi_lite_slave_enhanced.sv  ← Enhanced testbench (NEW)
│   └── ... other RTL
│
├── python/
│   └── host/
│       ├── axi_master_sim.py         ← Python AXI simulator (USED BY COCOTB)
│       └── axi_driver.py             ← High-level driver (USED BY COCOTB)
│
└── ... other project directories
"""

# ============================================================================
# SECTION 4: RUNNING TESTS
# ============================================================================

"""
Running with Different Simulators:

A. Using iverilog (default, simplest):
   $ cd /workspaces/ACCEL-v1/accel\ v1
   $ make -f tb/Makefile.cocotb SIM=iverilog
   
   Output:
   ==========================================
     AXI4-Lite Slave Testbench
   ==========================================
   
   >>> Suite 1: Single Writes (Valid Addresses)
   [TEST 1] WRITE addr=0x50 data=0x0000001
   ...

B. Using Verilator (faster, recommended for large designs):
   $ make -f tb/Makefile.cocotb SIM=verilator
   
C. Using VCS (if available):
   $ make -f tb/Makefile.cocotb SIM=vcs

D. Generate waveforms (VCD format for viewing in GTKWave):
   $ make -f tb/Makefile.cocotb trace
   
   View waveform:
   $ gtkwave tb/sim.vcd

E. Clean up:
   $ make -f tb/Makefile.cocotb clean
"""

# ============================================================================
# SECTION 5: UNDERSTANDING TEST OUTPUT
# ============================================================================

"""
When you run the testbench, you'll see output like:

==========================================
  AXI4-Lite Slave Testbench
==========================================

>>> Suite 1: Single Writes (Valid Addresses)

[TEST 1] WRITE addr=0x50 data=0x00000001
  [INFO] AW/W sent, waiting for response
  [PASS] Response=00

[TEST 2] WRITE addr=0x51 data=0x00000001
  [INFO] AW/W sent, waiting for response
  [PASS] Response=00

[CSR-W] STORED addr=0x50 data=0x00000001
[CSR-R] FETCHED addr=0x50 data=0x00000001

...

==========================================
  TEST SUMMARY
==========================================
  Total:  XX
  Passed: XX
  Failed: 0
  Status: ALL PASSED
==========================================

Key indicators:
- [PASS] = Test passed ✓
- [FAIL] = Test failed ✗
- Response=00 = OKAY (success)
- Response=11 = SLVERR (error)
- Status: ALL PASSED = All tests successful
"""

# ============================================================================
# SECTION 6: COCOTB TESTBENCH (Python Integration)
# ============================================================================

"""
The Cocotb testbench (cocotb_axi_master_test.py) provides:

1. Python-driven AXI stimulus:
   - Creates AXI transactions in Python
   - Drives Verilog DUT directly
   - Verifies responses

2. Test cases included:
   
   a) test_axi_write_single: Single write to valid address
   b) test_axi_read_single: Single read and verify
   c) test_axi_invalid_address: Error handling (invalid addr)
   d) test_axi_multiple_writes: Sequential transactions
   e) test_python_axi_integration: Deep integration test

3. Usage:
   
   To run specific test:
   $ make -f tb/Makefile.cocotb SIM=iverilog TESTS=test_axi_write_single
   
   To run all Cocotb tests:
   $ make -f tb/Makefile.cocotb SIM=iverilog
"""

# ============================================================================
# SECTION 7: RUNNING COCOTB TESTS (Step-by-Step)
# ============================================================================

"""
Step-by-step guide to run Cocotb tests:

1. Change to project directory:
   $ cd /workspaces/ACCEL-v1/accel\ v1

2. Verify Python paths are set:
   $ echo $PYTHONPATH
   # Should include path to python/host/

3. Run Cocotb simulation:
   $ make -f tb/Makefile.cocotb SIM=iverilog
   
   The build system will:
   ✓ Compile Verilog with iverilog
   ✓ Load Python Cocotb module
   ✓ Execute Python test functions
   ✓ Drive Verilog signals from Python
   ✓ Verify responses
   ✓ Generate report

4. Check results:
   - Console output shows test results
   - Check for "ALL PASSED" at end
   - Or check for any [FAIL] messages

5. Advanced: Generate waveforms:
   $ make -f tb/Makefile.cocotb TRACE=1 SIM=iverilog
   $ gtkwave tb/sim.vcd

6. To debug a failing test:
   a) Look at the [FAIL] message for what went wrong
   b) Add debug statements to cocotb_axi_master_test.py
   c) Rerun with: make -f tb/Makefile.cocotb clean && make -f tb/Makefile.cocotb
"""

# ============================================================================
# SECTION 8: INTEGRATING PYTHON AXI MASTER SIMULATOR
# ============================================================================

"""
How Python AXIMasterSim integrates with Cocotb:

Python Side (axi_master_sim.py):
├── AXIMasterSim class
├── Methods:
│   ├── write_single(addr, data) → (success, response)
│   ├── read_single(addr) → (data, response)
│   ├── write_burst(addr, data_list) → (success, beats, responses)
│   └── read_burst(addr, length) → (data_list, responses)
└── Storage:
    └── csr_memory dict (simulates FPGA registers)

Cocotb Test (cocotb_axi_master_test.py):
├── Imports AXIMasterSim
├── Creates instance: axi = AXIMasterSim()
├── Calls: axi.write_single(0x50, 0xDEADBEEF)
├── Gets response from Python (simulated)
├── Mirrors transaction to Verilog:
│   ├── dut.s_axi_awaddr = 0x50
│   ├── dut.s_axi_wdata = 0xDEADBEEF
│   └── ... (drive AXI signals)
├── Waits for Verilog response
└── Verifies both match

Flow:
┌─────────────────┐
│  Python Code    │  ← Test case
│  (Cocotb)       │
└────────┬────────┘
         │
    ┌────▼────────────────────┐
    │ AXIMasterSim Instance   │
    │ (Python simulation)     │
    │ - Validates address     │
    │ - Stores data           │
    │ - Returns response      │
    └────┬────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │ Mirror to Verilog AXI Bus         │
    │ - Drive signals                   │
    │ - Wait for response               │
    │ - Verify against Python           │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────┐
    │  Verilog DUT      │
    │  (axi_lite_slave) │
    │  - Process AXI    │
    │  - Store/read CSR │
    │  - Respond        │
    └────┬──────────────┘
         │
    ┌────▼────────────────┐
    │ Verify             │
    │ Python == Verilog? │
    └─────────────────────┘
"""

# ============================================================================
# SECTION 9: COMMON ISSUES & TROUBLESHOOTING
# ============================================================================

"""
Issue 1: "cocotb-config: command not found"
Solution:
  $ pip install cocotb

Issue 2: "Verilog source files not found"
Solution:
  Make sure you run from project root:
  $ cd /workspaces/ACCEL-v1/accel\ v1
  $ make -f tb/Makefile.cocotb

Issue 3: "ModuleNotFoundError: No module named 'axi_master_sim'"
Solution:
  Ensure PYTHONPATH includes python/host/:
  $ export PYTHONPATH=/workspaces/ACCEL-v1/accel\ v1/python/host:$PYTHONPATH
  $ make -f tb/Makefile.cocotb

Issue 4: Tests time out
Solution:
  a) Check for deadlocks in DUT
  b) Verify AXI handshaking is correct
  c) Check response generation logic
  d) Increase timeout in Makefile: TEST_TIMEOUT_US = 200

Issue 5: "Response mismatch" errors
Solution:
  a) Verify CSR address mapping is correct
  b) Check csr_valid_addrs set in axi_lite_slave.sv
  c) Verify write/read logic
  d) Check response code generation

Issue 6: Waveform viewer won't open
Solution:
  $ which gtkwave
  # If not found:
  $ apt-get install gtkwave
  # Then:
  $ gtkwave /path/to/sim.vcd
"""

# ============================================================================
# SECTION 10: NEXT STEPS
# ============================================================================

"""
After verifying AXI implementation:

1. Extend testbench with more test cases:
   - Partial writes (strb != 0xF)
   - Burst transfers
   - Back-to-back transactions
   - Protocol violations

2. Integrate with DMA bridge:
   - Test axi_dma_bridge.sv
   - Verify burst → FIFO conversion
   - Test data integrity

3. Full system integration:
   - Connect to systolic array
   - Test end-to-end data flow
   - Verify performance metrics

4. Synthesis:
   - Synthesize for FPGA (Quartus/Vivado)
   - Run on actual hardware
   - Verify against Python reference

"""

# ============================================================================
# SECTION 11: USEFUL COMMANDS
# ============================================================================

"""
Useful Make Targets:

make -f tb/Makefile.cocotb
  → Run simulation (default: iverilog)

make -f tb/Makefile.cocotb SIM=verilator
  → Run with Verilator (faster)

make -f tb/Makefile.cocotb clean
  → Clean artifacts

make -f tb/Makefile.cocotb list_tests
  → List available Cocotb tests

make -f tb/Makefile.cocotb trace
  → Run with waveform capture

make -f tb/Makefile.cocotb wave
  → Open waveform viewer

make -f tb/Makefile.cocotb help
  → Show help

Useful Python Commands:

# Run original testbench
cd /workspaces/ACCEL-v1/accel\ v1
python3 python/host/axi_master_sim.py

# Run AXI driver examples
python3 python/host/axi_driver.py

# Check metrics
python3 -c "
from python.host.axi_master_sim import AXIMasterSim
axi = AXIMasterSim(debug=True)
axi.write_single(0x50, 0xDEADBEEF)
axi.read_single(0x50)
axi.print_metrics()
"
"""

print(__doc__)
