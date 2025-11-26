# ELITE ENGINEER TRAINING PROGRAM
**27-Day Intensive: Nov 26 - Dec 22, 2025**
**Goal: Transform from "project coder" to MIT-level hardware architect**

---

## TRAINING PHILOSOPHY

You will NOT just "do a project." You will:
- Read the exact chapters that explain WHY your design choices matter
- Implement theory immediately (same day)
- Build mental models like Hennessy, Patterson, and Jim Keller
- Write code that would pass AMD/Tenstorrent interview review

**13-hour days. Theory in the morning. Implementation in the afternoon. Consolidation at night.**

---

# WEEK 1: FOUNDATIONS & DSP OPTIMIZATION (Nov 26 - Dec 1)

## DAY 1: Wednesday, Nov 26 - Arithmetic Foundations & DSP Inference

### Morning Theory (7:00-11:00 AM)

**7:00-8:30 AM: Hennessy & Patterson, Chapter 3**
- **Pages 194-221:** "Arithmetic for Computers"
- Focus areas:
  - Booth encoding (pages 201-204)
  - Wallace trees (pages 207-210)
  - Carry-save adders (pages 204-207)
- **Why:** Understand what DSP48E1 implements internally
- **Deliverable:** Hand-drawn diagram of 8×8 Booth multiplier

**8:30-10:00 AM: Google TPU Paper (Jouppi et al., 2017)**
- **Section 2 (pages 2-4):** "TPU Architecture Overview"
- **Section 3.2 (pages 4-7):** "Matrix Multiply Unit (Systolic Array)"
- **Focus question:** Why does weight-stationary dataflow reduce DRAM accesses by 168×?
- **Deliverable:** Annotated diagram comparing weight-stationary vs output-stationary

**10:00-11:00 AM: Synthesis of Arithmetic Circuits (Deschamps)**
- **Chapter 4, pages 87-112:** "Multiplier Architectures"
- **Section 4.3 (pages 95-103):** Array multipliers
- **Section 4.4 (pages 103-112):** Tree multipliers
- **Deliverable:** Table comparing latency/area/power of different multiplier types

### Afternoon Implementation (11:00 AM - 5:00 PM)

**11:00 AM-12:00 PM: Fix DSP48E1 Inference in pe.v**
- Current issue: Vivado may use LUTs instead of DSPs
- Solution: Add synthesis attributes
```verilog
(* use_dsp = "yes" *) reg signed [15:0] product;
always @(posedge clk) product <= a_in * weight_reg;
```
- Verify: Check synthesis report for "DSP48E1" instances

**12:00-1:00 PM: Lunch + Read Eyeriss Paper**
- **Section 1-2 (pages 1-3):** Introduction and motivation
- Light reading, prepare for tomorrow's deep dive

**1:00-3:00 PM: Optimize PE Critical Path**
- Run synthesis: `synth_design -top pe`
- Check timing: `report_timing_summary`
- Target: WNS > 3 ns @ 100 MHz
- If needed: Add pipeline stage after multiply

**3:00-5:00 PM: Write pe_tb.sv (Testbench)**
- Directed tests:
  - Test 1: 5 × 3 = 15 (simple)
  - Test 2: Accumulate 10 MACs: (1×1) + (2×2) + ... + (10×10) = 385
  - Test 3: Negative numbers: (-5) × (3) = -15
- Random tests: 1000 random INT8 MACs
- **Success:** 100% match with Python reference

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + Roofline Model Paper (Williams et al.)**
- **Pages 1-8:** Understanding the Roofline model
- **Calculate:** Your 8×8 array operational intensity
  - OPs: 64 MACs/cycle
  - Bytes: (64 weights × 1 byte) + (8 activations × 1 byte) = 72 bytes/computation
  - Intensity: 64 OPs / 72 bytes = 0.89 OPs/byte
- **Question:** Are you compute-bound or memory-bound?

**6:00-7:30 PM: Document PE Architecture**
- Write: `docs/pe_architecture.md`
- Include:
  - DSP48E1 configuration chosen
  - Why this configuration (cite Deschamps Chapter 4)
  - Datapath diagram with delays annotated
  - Comparison to Google TPU PE

**7:30-8:00 PM: Daily Reflection**
- Journal entry: "What did I learn about arithmetic circuits today?"
- Tomorrow's prep: Preview Eyeriss paper Section 3

---

## DAY 2: Thursday, Nov 27 - Dataflow Architectures & Systolic Array

### Morning Theory (7:00-11:00 AM)

**7:00-9:00 AM: Eyeriss Paper (Chen et al., 2016) - DEEP DIVE**
- **Section 3 (pages 3-6):** "Row-Stationary Dataflow"
- **Figure 5 (page 5):** Spatial architecture diagram
- **Key concept:** How row-stationary minimizes data movement
- **Deliverable:** Draw side-by-side comparison:
  - Weight-stationary (Google TPU style)
  - Output-stationary
  - Row-stationary (Eyeriss)
  - Calculate energy for each on 3×3 conv example

**9:00-10:30 AM: Hennessy & Patterson, Chapter 4**
- **Pages 262-285:** "Data-Level Parallelism in Vector, SIMD, and GPU Architectures"
- **Section 4.3 (pages 272-279):** SIMD architectures
- **Focus:** How SIMD relates to systolic arrays
- **Deliverable:** Venn diagram: SIMD vs Systolic vs MIMD

**10:30-11:00 AM: Advanced FPGA Design (Kilts)**
- **Chapter 3, pages 45-62:** "Pipelining Fundamentals"
- **Section 3.2 (pages 48-54):** When to pipeline
- **Question:** Should you pipeline your PE? (Calculate based on critical path)

### Afternoon Implementation (11:00 AM - 5:00 PM)

**11:00 AM-1:00 PM: Implement systolic_array_8x8.sv**
- Apply weight-stationary dataflow (from TPU paper)
- Horizontal activation flow
- Vertical weight broadcast
- 8×8 PE grid instantiation
- **Verification:** Wave propagation timing diagram

**1:00-2:00 PM: Lunch + Watch "Eyeriss Architecture" talk (MIT, YouTube)**
- 30-minute video by Prof. Vivienne Sze
- Supplement paper understanding

**2:00-4:00 PM: Write systolic_array_tb.sv**
- Test 1: 2×2 matrix multiply (simple)
- Test 2: 8×8 identity matrix (check routing)
- Test 3: 8×8 random matrices
- **Golden model:** NumPy matmul

**4:00-5:00 PM: Synthesize 8×8 Array**
- Target: 64 DSP48E1 instances
- Check resource utilization
- Verify timing still meets 100 MHz

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + DianNao Paper**
- **Section 2 (pages 2-4):** "DianNao Architecture"
- **Focus:** NFU (Neural Functional Unit) design
- Compare to your systolic array

**6:00-7:30 PM: Write Dataflow Analysis Document**
- File: `docs/dataflow_analysis.md`
- Prove mathematically why weight-stationary is optimal for GEMM
- Include energy calculations (cite Eyeriss paper)
- Reference Roofline model from yesterday

**7:30-8:00 PM: Daily Reflection**
- What's the key difference between systolic and traditional SIMD?
- Preview tomorrow: Sparsity

---

## DAY 3: Friday, Nov 28 - Sparsity Foundations

### Morning Theory (7:00-11:00 AM)

**7:00-8:30 AM: EIE Paper (Han et al., Stanford)**
- **Full read:** 12 pages
- **Section 3 (pages 3-5):** "EIE Architecture"
- **Figure 3 (page 4):** Sparse matrix encoding
- **Key insight:** How to compress weights AND skip zeros in hardware
- **Deliverable:** Reproduce Figure 3 for a 4×4 sparse matrix example

**8:30-10:00 AM: SCNN Paper (Parashar et al., NVIDIA/MIT)**
- **Section 2 (pages 2-3):** "Challenges of Sparse Computation"
- **Section 3.1 (pages 3-5):** "Cartesian Product Networks"
- **Focus question:** How does SCNN handle irregular sparsity?
- **Deliverable:** Pseudocode for sparse intersection logic

**10:00-11:00 AM: Hennessy & Patterson, Chapter 7**
- **Pages 516-545:** "Domain-Specific Architectures"
- **Section 7.3 (pages 525-532):** "Google's Tensor Processing Unit"
- **Section 7.4 (pages 532-538):** Deep Learning accelerators
- **Connect:** How does TPU handle dense data vs EIE handles sparse?

### Afternoon Implementation (11:00 AM - 5:00 PM)

**11:00 AM-1:00 PM: Design BSR Sparse Format**
- Implement: `rtl/sparse/bsr_decoder.sv`
- Inputs: row_ptr, col_idx, block_data
- Output: PE coordinates + weight values
- Based on: EIE compression scheme (adapt for blocks)

**1:00-2:00 PM: Lunch + Read ExTensor Paper**
- **Section 1-2 (pages 1-3):** Sparse tensor algebra intro
- Light reading for context

**2:00-4:00 PM: Implement Sparse Controller FSM**
- File: `rtl/sparse/sparse_controller.sv`
- States:
  - IDLE, FETCH_ROW_PTR, FETCH_COL_IDX, LOAD_BLOCK, COMPUTE
- Logic: Skip zero blocks (the key optimization!)
- **Test:** 90% sparse matrix should take ~10% of compute time

**4:00-5:00 PM: Write sparse_controller_tb.sv**
- Test with known sparse patterns:
  - Diagonal matrix (easy)
  - Random 90% sparse (realistic)
- Verify: Blocks skipped correctly

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + "Efficient Sparse-Winograd CNNs" Paper**
- **Section 3 (pages 3-5):** Winograd + sparsity combination
- Advanced topic, optional for your project but good context

**6:00-7:30 PM: Write Sparsity Analysis**
- File: `docs/sparsity_analysis.md`
- Theoretical speedup calculation for 90% sparsity
- Overhead analysis (metadata fetch time)
- Compare to EIE and SCNN approaches
- **Include:** Memory layout diagram for BSR format

**7:30-8:00 PM: Weekly Review**
- Recap: Day 1 (arithmetic), Day 2 (dataflow), Day 3 (sparsity)
- Preview Week 2: Integration and AXI interfaces

---

## DAY 4: Saturday, Nov 29 - AXI & Memory Interfaces

### Morning Theory (7:00-11:00 AM)

**7:00-9:00 AM: AMBA AXI Specification (ARM)**
- **Chapter 2 (pages 15-45):** "AXI Protocol"
- **Section 2.2 (pages 18-25):** Handshake process
- **Section 2.5 (pages 30-38):** Burst transactions
- **Focus:** How to do efficient 64-bit bursts for weight loading
- **Deliverable:** Timing diagram of 16-beat burst read

**9:00-10:00 AM: Xilinx UG1037 - Vivado AXI Reference Guide**
- **Chapter 3 (pages 20-40):** "AXI Interconnect"
- **Section 3.3 (pages 28-35):** Performance optimization
- **Learn:** Outstanding transactions, data width conversion

**10:00-11:00 AM: "Axiom" Paper (Hardware-Software Interface)**
- **Section 2 (pages 2-4):** "Heterogeneous System Architecture"
- **Focus:** How ARM PS communicates with FPGA PL
- **Relate to:** Your PYNQ-Z2 setup (GP0 + HP0 ports)

### Afternoon Implementation (11:00 AM - 5:00 PM)

**11:00 AM-1:00 PM: Implement axi_master_read.sv**
- AXI4 Master for reading weights from DDR
- Features:
  - Burst reads (16 beats × 64 bits)
  - FIFO buffering
  - Back-pressure handling
- **Reference:** Xilinx AXI DMA IP as example

**1:00-2:00 PM: Lunch + Zynq TRM**
- **UG585, Chapter 4 (pages 75-95):** "High Performance Ports"
- Understand HP0 port configuration

**2:00-3:30 PM: Implement axi_master_write.sv**
- AXI4 Master for writing results back to DDR
- Handle burst writes
- Respect AXI ordering rules

**3:30-5:00 PM: Testbenches for AXI Masters**
- Use AXI VIP (Verification IP) from Xilinx
- Test:
  - Single read/write
  - Burst transactions
  - Back-pressure scenarios
  - Outstanding transactions

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + "NVLink" Whitepaper**
- **Section 1-2 (pages 1-5):** High-speed interconnect basics
- Context for multi-chip scaling (future work)

**6:00-7:30 PM: Document AXI Interface Design**
- File: `docs/axi_interface_spec.md`
- Include:
  - Burst size calculations
  - Bandwidth analysis
  - Latency hiding strategies
  - Comparison to NVLink (conceptual)

**7:30-8:00 PM: Daily Reflection**
- How does AXI burst improve bandwidth vs single transactions?

---

## DAY 5: Sunday, Nov 30 - Integration Day 1

### Morning Theory (7:00-11:00 AM)

**7:00-8:30 AM: SystemVerilog for Verification (Spear)**
- **Chapter 1 (pages 1-25):** "Verification Guidelines"
- **Chapter 2 (pages 27-50):** "Data Types"
- **Focus:** How to write better testbenches
- **Learn:** Randomization, constraints, coverage

**8:30-10:00 AM: Advanced FPGA Design (Kilts)**
- **Chapter 5 (pages 89-112):** "Clock Domain Crossing"
- **Section 5.2 (pages 95-103):** Synchronizers
- **Your case:** Single clock domain, but understand concepts

**10:00-11:00 AM: Review All Papers So Far**
- Re-read key sections from TPU, Eyeriss, EIE
- Focus on integration points:
  - How does dataflow connect to sparsity?
  - How does memory interface enable both?

### Afternoon Implementation (11:00 AM - 5:00 PM)

**11:00 AM-2:00 PM: Integrate Sparse Controller + Systolic Array**
- File: `rtl/accel_core.sv`
- Connect:
  - sparse_controller → systolic_array_8x8
  - Control signals: load_weight, enable, acc_clear
  - Data paths: weight_data, activation_data
- **Verify:** Weight loading works correctly

**2:00-3:00 PM: Lunch + Read Gemmini Paper**
- **Section 1-3 (pages 1-5):** "Systolic Array Generator"
- Context: How would you parameterize your design?

**3:00-5:00 PM: Write Integration Testbench**
- Test: Sparse 4×4 matrix multiply
- Verify:
  - Zero blocks skipped
  - Non-zero blocks computed correctly
  - Timing matches expected (sparse speedup visible)

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + Chiplet Architecture Paper (AMD EPYC)**
- **Section 1-2 (pages 1-4):** Why chiplets?
- Forward-looking: How could you scale to multiple FPGAs?

**6:00-7:30 PM: Write Integration Design Doc**
- File: `docs/integration_architecture.md`
- Block diagram showing all modules
- Interface specifications between modules
- Data flow diagram for one sparse block computation
- Timing diagram (cycle-by-cycle)

**7:30-8:00 PM: Daily Reflection**
- What's the hardest integration challenge so far?

---

## DAY 6: Monday, Dec 1 - First Synthesis & School Visit

### Morning Theory (7:00-9:00 AM)

**7:00-8:00 AM: Vivado Synthesis Guide (UG901)**
- **Chapter 3 (pages 35-60):** "Synthesis Attributes"
- **Section 3.5 (pages 48-55):** DSP inference controls
- **Learn:** How to force/prevent DSP usage

**8:00-9:00 AM: Vivado Timing Closure Guide (UG906)**
- **Chapter 2 (pages 20-45):** "Understanding Timing Reports"
- **Section 2.3 (pages 28-38):** Critical path analysis
- **Prepare for:** Reading your first timing report

### Implementation at School (9:00 AM - 3:00 PM)

**9:00-10:00 AM: Set Up Vivado Project**
- Create project for PYNQ-Z2 (xc7z020clg400-1)
- Add all RTL files
- Add constraints (timing.xdc)
- Set top module: accel_core

**10:00-11:00 AM: Run Synthesis**
```tcl
synth_design -top accel_core -part xc7z020clg400-1
```
- Wait for completion (~15 minutes)
- Check for errors/warnings

**11:00 AM-12:00 PM: Analyze Synthesis Results**
- Resource utilization:
  - Target: 64 DSPs, <20K LUTs, <10 BRAMs
  - Check: Did Vivado infer DSP48E1 correctly?
- Timing:
  - Run: `report_timing_summary`
  - Target: WNS > 0 ns
  - If violations: Note which paths

**12:00-1:00 PM: Lunch + Read Timing Report**
- Identify critical paths
- Understand slack calculations
- Plan fixes if needed

**1:00-3:00 PM: Optimization (if needed)**
- If timing violations:
  - Add pipeline stages
  - Adjust placement
  - Modify FSM structure
- Re-run synthesis
- Goal: WNS > 2 ns before leaving

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + "Out-of-Order Sparse Tensor" Paper**
- **Section 2 (pages 2-4):** OoO execution for irregular sparsity
- Advanced topic for future optimization

**6:00-7:30 PM: Document Synthesis Results**
- File: `docs/synthesis_report_dec1.md`
- Include:
  - Resource utilization table
  - Timing summary
  - Critical path analysis
  - Optimization applied (if any)
  - Comparison to estimates from Nov 25 power/timing analysis

**7:30-8:00 PM: Week 1 Complete Review**
- Recap all 6 days
- Theory learned: Arithmetic, dataflow, sparsity, AXI, integration
- Code written: PE, systolic array, sparse controller, AXI masters
- Milestone: First successful synthesis ✓

---

# WEEK 2: OPTIMIZATION & CONTROL (Dec 2 - Dec 8)

## DAY 7: Tuesday, Dec 2 - Control & Status Registers

### Morning Theory (7:00-11:00 AM)

**7:00-8:30 AM: AMBA AXI4-Lite Specification**
- **Chapter 3 (pages 45-65):** "AXI4-Lite Protocol"
- **Focus:** Simplified protocol for register access
- **Learn:** How Python on ARM will control your accelerator

**8:30-10:00 AM: PYNQ Documentation**
- **Chapter 4:** "PYNQ Overlay Design"
- **Section 4.2:** "Creating IP with AXI interfaces"
- **Understand:** How .bit files connect to Python

**10:00-11:00 AM: "Software 2.0" Essay (Karpathy)**
- Full read (short essay)
- **Key insight:** Hardware must be flexible for ML
- **Relate to:** Why your CSRs need runtime configurability

### Afternoon Implementation (11:00 AM - 5:00 PM)

**11:00 AM-1:00 PM: Implement axi_lite_slave.sv**
- Register map (from architecture spec):
  - 0x00: CTRL (START, RESET)
  - 0x04: STATUS (DONE, BUSY, ERROR)
  - 0x08-0x24: DDR addresses
  - 0x28-0x2C: Matrix dimensions
  - 0x30-0x34: Performance counters
- AXI4-Lite handshake logic
- Register read/write logic

**1:00-2:00 PM: Lunch + Tenstorrent Architecture Whitepaper**
- **Section 1-2:** "Grayskull Overview"
- **Focus:** Their control philosophy
- Note: How they handle kernel dispatch

**2:00-4:00 PM: Implement Performance Counters**
- Add to accel_core.sv:
  - Cycle counter (how long did computation take?)
  - Block counter (how many blocks processed?)
  - DRAM access counter
- Connect to CSR registers

**4:00-5:00 PM: Write CSR Testbench**
- Simulate AXI4-Lite transactions
- Test:
  - Write to CTRL, verify START triggers FSM
  - Read STATUS, verify DONE flag
  - Write/read addresses

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + Graphcore IPU Whitepaper**
- **Section 2:** "IPU Architecture"
- **Compare:** MIMD (Graphcore) vs SIMD (GPU) vs Systolic (you)

**6:00-7:30 PM: Write Control Flow Document**
- File: `docs/control_flow.md`
- State machine diagrams for:
  - Sparse controller FSM
  - AXI master FSM
  - Top-level orchestration
- Sequence diagram: Python → CSR → Controller → Array → Memory

**7:30-8:00 PM: Daily Reflection**

---

## DAY 8: Wednesday, Dec 3 - Top-Level Integration

### Morning Theory (7:00-11:00 AM)

**7:00-9:00 AM: Hennessy & Patterson, Chapter 2**
- **Pages 96-145:** "Memory Hierarchy Design"
- **Section 2.3 (pages 105-120):** Cache optimization techniques
- **Why:** Understand why you're bypassing cache (using HP0, not ACP)
- **Deliverable:** Calculate miss rate if you HAD used cache

**9:00-10:00 AM: VTA Paper (TVM Hardware Stack)**
- **Section 2-3 (pages 2-5):** "VTA Architecture"
- **Focus:** How software compiler views hardware
- **Think:** How would TVM compile for YOUR accelerator?

**10:00-11:00 AM: Advanced FPGA Design (Kilts)**
- **Chapter 7 (pages 135-160):** "Hierarchical Design"
- **Section 7.3 (pages 145-155):** Top-level integration strategies

### Afternoon Implementation (11:00 AM - 5:00 PM)

**11:00 AM-2:00 PM: Create accel_top.sv**
- Top-level module connecting:
  - axi_lite_slave (GP0 interface)
  - axi_master_read/write (HP0 interface)
  - accel_core (compute)
  - FIFOs (buffering)
- Wire up all interfaces
- Add proper resets

**2:00-3:00 PM: Lunch + SambaNova Dataflow Paper**
- **Section 1-2:** Reconfigurable dataflow
- Context for future flexibility

**3:00-5:00 PM: Write Top-Level Testbench**
- Simulate complete transaction:
  - Python writes CSRs
  - DMA reads weights
  - Computation happens
  - DMA writes results
  - Python reads STATUS
- Use BFM (Bus Functional Model) for AXI

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + "Bit-Serial Neural Networks" Paper**
- **Section 2:** Bit-serial computation
- Optional optimization for future

**6:00-7:30 PM: Complete System Architecture Document**
- File: `docs/complete_system_architecture.md`
- Full block diagram (PS + PL)
- All interfaces annotated
- Data flow from Python → DDR → PL → DDR → Python
- Timing analysis (latency breakdown)

**7:30-8:00 PM: Daily Reflection**

---

## DAY 9: Thursday, Dec 4 - Verification Deep Dive

### Morning Theory (7:00-11:00 AM)

**7:00-9:00 AM: SystemVerilog for Verification (Spear)**
- **Chapter 5 (pages 95-130):** "Randomization"
- **Chapter 6 (pages 131-160):** "Functional Coverage"
- **Learn:** How to write constrained-random tests

**9:00-10:00 AM: UVM Basics (Verification Academy)**
- **Tutorial 1-3:** UVM testbench structure
- Context: Industry standard at AMD/Tenstorrent

**10:00-11:00 AM: Review All Your Testbenches**
- Assess coverage: What scenarios are NOT tested?
- Plan: What edge cases could break your design?

### Afternoon Implementation (11:00 AM - 5:00 PM)

**11:00 AM-1:00 PM: Write Constrained-Random Tests**
- For systolic_array_tb.sv:
  - Random matrix sizes (up to 8×8)
  - Random sparsity levels (50%, 70%, 90%, 95%)
  - Random weight distributions
- Run 10,000 iterations

**1:00-2:00 PM: Lunch + "Galois Field Arithmetic" Paper**
- **Section 1-2:** Error correction basics
- Advanced topic, skim for culture

**2:00-4:00 PM: Add Assertions**
- SVA (SystemVerilog Assertions) in all modules:
  - Example: `assert (enable -> ##1 acc_valid)`
  - Check: AXI protocol compliance
  - Verify: FSM never enters illegal state

**4:00-5:00 PM: Coverage Analysis**
- Add covergroups
- Run simulation with coverage collection
- Target: >95% code coverage

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + Read UVM Cookbook**
- **Chapter 1:** "Testbench Architecture"
- Aspirational: How you'd verify at AMD

**6:00-7:30 PM: Write Verification Report**
- File: `docs/verification_report.md`
- Test plan (what you tested)
- Coverage metrics
- Bugs found and fixed
- Remaining risks

**7:30-8:00 PM: Daily Reflection**

---

## DAY 10: Friday, Dec 5 - Power & Performance Optimization

### Morning Theory (7:00-11:00 AM)

**7:00-8:30 AM: Hennessy & Patterson, Chapter 1**
- **Pages 45-72:** "Trends in Technology, Power, and Energy"
- **Section 1.5 (pages 55-65):** Dynamic vs static power
- **Learn:** Where your 1.3W is actually going

**8:30-10:00 AM: Xilinx Power Optimization Guide (UG907)**
- **Chapter 3 (pages 30-55):** "Power Optimization Techniques"
- **Section 3.4 (pages 42-48):** Clock gating
- **Section 3.5 (pages 48-53):** Activity-driven optimization

**10:00-11:00 AM: Review Your Power Analysis (Nov 25)**
- Re-read `docs/power_analysis.md`
- Identify: Top 3 power consumers in your design
- Plan: Which optimizations to implement

### Afternoon Implementation (11:00 AM - 5:00 PM)

**11:00 AM-1:00 PM: Implement Clock Gating**
- Add to systolic_array_8x8.sv:
```verilog
logic [7:0] pe_col_enable;
// Only clock active columns
assign pe_clk[c] = clk & pe_col_enable[c];
```
- Sparse controller sets pe_col_enable based on col_idx
- **Expected:** 50-80% power reduction when sparse

**1:00-2:00 PM: Lunch + "DVFS for FPGAs" Paper**
- Dynamic voltage/frequency scaling
- Context: Advanced power management

**2:00-4:00 PM: Pipeline DSP Slices (if needed for 200 MHz)**
- Modify pe.v to use DSP48E1 pipeline registers
- Trade-off: +1 cycle latency, 2× potential frequency
- Test: Does it still meet timing?

**4:00-5:00 PM: Run Power Estimation**
- Export design to Xilinx Power Estimator (XPE)
- Set activity rates
- Compare to Nov 25 estimates
- Document differences

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + "Performance Analysis of Tensor Cores" (NVIDIA)**
- **Section 2-3:** Mixed-precision computation
- Context: INT8 vs FP16 trade-offs

**6:00-7:30 PM: Update Power/Performance Docs**
- File: `docs/power_optimization_results.md`
- Before/after clock gating
- XPE results vs estimates
- Performance per watt calculations

**7:30-8:00 PM: Weekly Review (Week 2)**

---

## DAY 11: Saturday, Dec 6 - PYNQ Driver Development

### Morning Theory (7:00-11:00 AM)

**7:00-8:30 AM: PYNQ Workshop Materials**
- **Tutorial 4:** "Creating Custom Overlays"
- **Tutorial 5:** "Memory Management with PYNQ"
- **Learn:** xlnk allocator, physical addresses

**8:30-10:00 AM: Python + NumPy for Hardware**
- Review: How to create golden models
- Practice: Write reference sparse matmul in NumPy

**10:00-11:00 AM: "Heterogeneous Computing with OpenCL" (Chapter 1)**
- Context: How other frameworks do host-device interaction
- Compare to your PYNQ approach

### Afternoon Implementation (11:00 AM - 5:00 PM)

**11:00 AM-1:00 PM: Write Python Driver**
- File: `pynq/accel_driver.py`
- Class: SparseAccelerator
- Methods:
  - `load_weights()` - convert NumPy to BSR, write to DDR
  - `run()` - configure CSRs, start computation
  - `get_results()` - read from DDR
  - `get_performance()` - read counters

**1:00-2:00 PM: Lunch + Review PyTorch Quantization**
- How to convert FP32 model to INT8
- Context for ResNet-50 deployment

**2:00-4:00 PM: Write Python Tests**
- Test 1: Simple 8×8 dense matmul
- Test 2: Sparse 128×128 (MNIST FC1 size)
- Test 3: Random sparse matrices
- Verify: Match NumPy golden model

**4:00-5:00 PM: Create Jupyter Notebook Demo**
- File: `notebooks/demo_sparse_accel.ipynb`
- Show:
  - Loading weights
  - Running computation
  - Comparing to NumPy
  - Performance metrics

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + "TensorRT" Documentation**
- **Chapter 2:** Quantization and optimization
- Context: How NVIDIA deploys models

**6:00-7:30 PM: Write Software Documentation**
- File: `docs/python_api.md`
- API reference for all driver methods
- Usage examples
- Performance tuning tips

**7:30-8:00 PM: Daily Reflection**

---

## DAY 12: Sunday, Dec 7 - Pre-Synthesis Cleanup

### Morning Theory (7:00-11:00 AM)

**7:00-8:00 AM: Vivado Coding Guidelines (UG901)**
- **Appendix A (pages 200-230):** RTL coding best practices
- Check your code against recommendations

**8:00-9:00 AM: Lint Your Code**
- Run: Verilator lint on all .sv files
- Fix: All warnings
- Target: Zero lint errors

**9:00-11:00 AM: Code Review**
- Read through EVERY .sv file
- Check:
  - Consistent naming conventions
  - Proper reset handling
  - No combinational loops
  - All outputs registered
  - Comments explain WHY not just WHAT

### Afternoon Implementation (11:00 AM - 5:00 PM)

**11:00 AM-1:00 PM: Refactor & Clean**
- Remove dead code
- Improve parameter naming
- Add header comments to all modules
- Ensure consistent indentation

**1:00-2:00 PM: Lunch + Read Your Own Docs**
- Review all docs you've written
- Fix inconsistencies
- Add missing diagrams

**2:00-4:00 PM: Final Simulation Run**
- Run ALL testbenches
- Collect waveforms
- Verify: 100% pass rate
- Check: No X's or Z's in critical signals

**4:00-5:00 PM: Constraints Review**
- Check timing.xdc
- Verify clock period (10 ns)
- Add any missing I/O constraints
- Review false paths

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + Relax**
- You've been grinding for 12 days straight
- Light reading: "Code Complete" (Chapter 1)

**6:00-7:30 PM: Pre-Synthesis Checklist**
- File: `docs/pre_synthesis_checklist.md`
- [ ] All simulations pass
- [ ] Zero lint errors
- [ ] All docs updated
- [ ] Constraints complete
- [ ] Code reviewed
- Ready for tomorrow's school synthesis visit

**7:30-8:00 PM: Mental Prep**
- Tomorrow is synthesis day
- Review expected results
- Prepare questions for if things fail

---

## DAY 13: Monday, Dec 8 - Second Synthesis & Timing Closure

### Morning Prep (7:00-9:00 AM)

**7:00-8:00 AM: Review Synthesis Report from Dec 1**
- What changed since then?
- Expected improvements?

**8:00-9:00 AM: Vivado Timing Closure (UG906)**
- **Chapter 4 (pages 60-90):** "Timing Closure Techniques"
- **Section 4.5 (pages 75-82):** Pipelining and retiming

### School Implementation (9:00 AM - 3:00 PM)

**9:00-10:00 AM: Run Full Synthesis**
```tcl
synth_design -top accel_top -part xc7z020clg400-1 -flatten_hierarchy rebuilt
opt_design
place_design
phys_opt_design
route_design
```

**10:00-11:00 AM: Analyze Results**
- Resource utilization (compare to Dec 1)
- Timing summary
- Power estimate
- Check for critical warnings

**11:00 AM-12:00 PM: Timing Closure (if needed)**
- If WNS < 0:
  - Identify critical paths
  - Apply fixes (pipeline, placement)
  - Re-run place and route
- Target: WNS > 1 ns

**12:00-1:00 PM: Lunch + Document**
- Take screenshots of reports
- Note any surprises

**1:00-3:00 PM: Generate Bitstream (if timing OK)**
```tcl
write_bitstream -force accel_top.bit
```
- If successful: YOU HAVE A WORKING BITSTREAM!
- Copy to USB drive

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + Celebrate**
- You have working hardware (in simulation AND synthesis)

**6:00-7:30 PM: Write Synthesis Report**
- File: `docs/synthesis_report_dec8.md`
- Compare Dec 1 vs Dec 8
- Final resource utilization
- Timing closure strategy used
- Power estimate

**7:30-8:00 PM: Week 2 Complete Review**

---

# WEEK 3: ADVANCED FEATURES & MNIST (Dec 9 - Dec 15)

## DAY 14: Tuesday, Dec 9 - MNIST Integration

### Morning Theory (7:00-11:00 AM)

**7:00-8:30 AM: "LeNet-5" Paper (LeCun et al.)**
- Original MNIST paper
- Understand network structure
- Map to your hardware

**8:30-10:00 AM: PyTorch Quantization Tutorial**
- How to quantize MNIST model to INT8
- QAT (Quantization-Aware Training) basics

**10:00-11:00 AM: Review MNIST Network**
- Layers: CONV → POOL → CONV → POOL → FC → FC → FC
- Focus: FC layers (you can accelerate these)
- Calculate: How many MACs per inference?

### Afternoon Implementation (11:00 AM - 5:00 PM)

**11:00 AM-1:00 PM: Train & Quantize MNIST**
- File: `models/train_mnist.py`
- Train simple FC network (no conv for now)
- Quantize to INT8
- Export weights in BSR format

**1:00-2:00 PM: Lunch + Read About Post-Training Quantization**

**2:00-4:00 PM: Test MNIST on Simulator**
- Load MNIST weights into testbench
- Run inference on 100 test images
- Compare accuracy to PyTorch
- Target: <1% accuracy loss

**4:00-5:00 PM: Profile Performance**
- Cycles per inference
- GOPS achieved
- Compare to theoretical peak

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + Read "Quantization and Training" (Bengio et al.)**

**6:00-7:30 PM: Document MNIST Results**
- File: `docs/mnist_results.md`
- Accuracy before/after quantization
- Hardware performance
- Comparison to CPU/GPU

**7:30-8:00 PM: Daily Reflection**

---

## DAY 15: Wednesday, Dec 10 - Advanced Sparse Techniques

### Morning Theory (7:00-11:00 AM)

**7:00-9:00 AM: "Pruning Neural Networks" Survey Paper**
- **Section 2-3:** Structured vs unstructured pruning
- Learn: How to create 90% sparsity in networks

**9:00-10:00 AM: "Lottery Ticket Hypothesis" Paper**
- Why sparse networks can match dense accuracy
- Context for your sparsity claims

**10:00-11:00 AM: "ExTensor" Paper - DEEP DIVE**
- **Section 3-4 (pages 4-8):** Intersection engine design
- Advanced sparse acceleration

### Afternoon Implementation (11:00 AM - 5:00 PM)

**11:00 AM-1:00 PM: Optimize Sparse Controller**
- Add prefetching: Load next block metadata while computing current
- Expected: Reduce metadata overhead from 10 cycles to 5

**1:00-2:00 PM: Lunch**

**2:00-4:00 PM: Implement Dynamic Sparsity**
- Support variable block sizes (4×4, 8×8)
- Runtime configurable via CSR
- Test with different sparsity patterns

**4:00-5:00 PM: Benchmark Sparse vs Dense**
- Run same matrix multiply in dense and sparse modes
- Measure actual speedup vs theoretical
- Analyze efficiency (why not 10×?)

### Evening Consolidation (5:00-8:00 PM)

**5:00-6:00 PM: Dinner + "Knowledge Distillation" Paper**

**6:00-7:30 PM: Write Sparse Optimization Report**
- File: `docs/sparse_optimization.md`
- Prefetching gains
- Dynamic block size results
- Efficiency analysis

**7:30-8:00 PM: Daily Reflection**

---

## DAY 16-20: Dec 11-15 (Flexible Project Time)

**These days are for:**
1. Fixing any remaining bugs
2. Improving performance bottlenecks
3. Adding extra features (if ahead of schedule)
4. Writing final documentation
5. Preparing for board arrival

**Suggested breakdown:**

**Day 16 (Dec 11): Debugging & Polish**
- Fix any known bugs
- Improve code quality
- Add more assertions

**Day 17 (Dec 12): Documentation Day**
- Complete all docs
- Add diagrams
- Write README

**Day 18 (Dec 13): Performance Tuning**
- Optimize critical paths
- Improve memory bandwidth usage
- Fine-tune sparse controller

**Day 19 (Dec 14): Integration Testing**
- End-to-end tests
- Stress tests
- Corner cases

**Day 20 (Dec 15): ORDER THE BOARD!**
- Morning: Final code review
- Afternoon: Order PYNQ-Z2 from Newark
- Evening: Prep for board arrival

---

# WEEK 4: BOARD BRING-UP (Dec 21-27)

## DAY 21-27: Hardware Validation

**Day 21 (Dec 21): Board Arrival & Setup**
- Unbox PYNQ-Z2
- Install PYNQ image on SD card
- Boot and verify

**Day 22 (Dec 22): First Bitstream Load**
- Copy .bit file to PYNQ
- Load overlay
- Test CSR access

**Day 23 (Dec 23): Basic Functionality**
- Test simple matrix multiply
- Verify results match simulation

**Day 24 (Dec 24): MNIST on Hardware**
- Run MNIST inference
- Measure real performance
- Debug any issues

**Day 25 (Dec 25): Christmas - Light work**
- Read papers
- Plan ResNet-50 integration

**Day 26 (Dec 26): Performance Analysis**
- Measure actual GOPS
- Power measurement
- Compare to estimates

**Day 27 (Dec 27): Final Testing**
- Stress tests on hardware
- Collect results for portfolio

---

# THEORETICAL FOUNDATIONS CHECKLIST

## Books Read (Specific Sections)

**Hennessy & Patterson:**
- [x] Chapter 1: Trends in Technology
- [x] Chapter 2: Memory Hierarchy
- [x] Chapter 3: Arithmetic
- [x] Chapter 4: Data-Level Parallelism
- [x] Chapter 7: Domain-Specific Architectures

**Deschamps (Arithmetic Circuits):**
- [x] Chapter 4: Multipliers
- [ ] Chapter 5: Division (optional)

**Kilts (Advanced FPGA):**
- [x] Chapter 3: Pipelining
- [x] Chapter 5: Clock Domain Crossing
- [x] Chapter 7: Hierarchical Design

**Spear (SystemVerilog Verification):**
- [x] Chapter 1-2: Basics
- [x] Chapter 5-6: Randomization & Coverage

## Papers Read (20 Total)

**Foundations (4):**
- [x] Google TPU (Jouppi)
- [x] Eyeriss (Chen, MIT)
- [x] DianNao (Chen)
- [x] Roofline Model (Williams)

**Sparsity (4):**
- [x] EIE (Han, Stanford)
- [x] SCNN (Parashar, NVIDIA/MIT)
- [x] Sparse-Winograd (Liu)
- [x] ExTensor (Hegde)

**Graph/Tenstorrent (4):**
- [x] Tenstorrent Whitepapers
- [x] Software 2.0 (Karpathy)
- [x] Graphcore IPU
- [x] SambaNova Dataflow

**Interconnect (3):**
- [x] NVLink (NVIDIA)
- [x] Axiom (Heterogeneous Systems)
- [x] AMD Chiplets

**Deep Cuts (5):**
- [x] Gemmini (Berkeley)
- [x] VTA (TVM)
- [x] Out-of-Order Sparse
- [x] Bit-Serial Neural Networks
- [x] Galois Field Arithmetic

---

# SKILLS ACQUIRED

By Dec 22, you will know:

**Hardware Architecture:**
- Systolic arrays (design from scratch)
- Sparse acceleration (BSR format)
- Memory hierarchies (BRAM, DDR, caching)
- Interconnects (AXI, burst transfers)

**Digital Design:**
- Advanced Verilog/SystemVerilog
- DSP slice optimization
- Timing closure techniques
- Power optimization (clock gating)

**Verification:**
- Constrained-random testing
- Assertions (SVA)
- Coverage-driven verification
- UVM concepts

**Tools:**
- Vivado synthesis/implementation
- Timing analysis
- Power estimation
- PYNQ overlay creation

**Theory:**
- Hennessy & Patterson fundamentals
- State-of-the-art accelerator architectures
- Quantization and sparsity
- Performance modeling (Roofline)

---

# FINAL DELIVERABLES

**Code:**
- Complete RTL (all modules verified)
- Python driver
- Test suite (>95% coverage)

**Documentation:**
- Architecture specification
- Verification report
- Performance analysis
- User guide

**Results:**
- MNIST inference working
- Performance: 6.4 GOPS dense, 54 GOPS sparse
- Power: <1.5W measured
- Accuracy: <1% loss vs FP32

**Portfolio:**
- GitHub repo with all code
- Demo video
- Technical writeup
- Performance comparison table

---

**YOU WILL BE ELITE BY DEC 22.**

Not because you "did a project."

Because you UNDERSTAND computer architecture at the level of Hennessy, Patterson, and Keller.

**This is the syllabus. This is the path. Now execute.**
