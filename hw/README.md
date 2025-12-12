# ACCEL-v1 Hardware Architecture

> 16×16 Weight-Stationary Systolic Array with BSR Sparse Acceleration

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ZYNQ ULTRASCALE+ / Z7020                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │              │     │                 ACCEL-v1 (PL)                   │   │
│  │   ARM PS     │     │  ┌───────────┐  ┌───────────┐  ┌────────────┐  │   │
│  │              │     │  │           │  │           │  │            │  │   │
│  │  ┌────────┐  │ AXI │  │  BSR DMA  │─▶│  Weight   │─▶│  16×16     │  │   │
│  │  │ Linux  │  │ HP  │  │  Engine   │  │  Buffer   │  │  Systolic  │  │   │
│  │  │ Driver │◀─┼─────┼─▶│           │  │  (BRAM)   │  │  Array     │  │   │
│  │  └────────┘  │     │  └───────────┘  └───────────┘  │            │  │   │
│  │              │     │                                 │   ┌────┐   │  │   │
│  │  ┌────────┐  │ AXI │  ┌───────────┐  ┌───────────┐  │   │ PE │×  │  │   │
│  │  │ Python │  │Lite │  │    CSR    │  │Activation │─▶│   └────┘   │  │   │
│  │  │ PYNQ   │◀─┼─────┼─▶│  Control  │  │  Buffer   │  │    256     │  │   │
│  │  └────────┘  │     │  │           │  │  (BRAM)   │  │            │──┼───┼──▶ Output
│  │              │     │  └───────────┘  └───────────┘  └────────────┘  │   │
│  └──────────────┘     │                                                 │   │
│                       │  ┌───────────┐  ┌───────────────────────────┐  │   │
│        DDR4           │  │   BSR     │  │    Output Accumulator     │  │   │
│   ┌─────────────┐     │  │ Scheduler │  │    + ReLU + Quantize      │  │   │
│   │ Weights     │     │  │           │  │                           │  │   │
│   │ Activations │◀────┼──│           │◀─│    (INT32 → INT8)         │  │   │
│   │ Results     │     │  └───────────┘  └───────────────────────────┘  │   │
│   └─────────────┘     └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔲 16×16 Systolic Array (Weight-Stationary)

```
                    Activations (broadcast down columns)
                    ↓     ↓     ↓     ↓           ↓
              ┌─────┬─────┬─────┬─────┬─ ─ ─┬─────┐
              │a[0] │a[1] │a[2] │a[3] │     │a[15]│
              └──┬──┴──┬──┴──┬──┴──┬──┴─ ─ ─┴──┬──┘
                 ↓     ↓     ↓     ↓           ↓
    ┌────┐    ┌─────┬─────┬─────┬─────┬─────┬─────┐
    │w[0]│───▶│PE   │PE   │PE   │PE   │ ... │PE   │───▶ psum[0]
    └────┘    │0,0  │0,1  │0,2  │0,3  │     │0,15 │
              └──┬──┴──┬──┴──┬──┴──┬──┴─────┴──┬──┘
    ┌────┐       ↓     ↓     ↓     ↓           ↓
    │w[1]│───▶┌─────┬─────┬─────┬─────┬─────┬─────┐
    └────┘    │PE   │PE   │PE   │PE   │ ... │PE   │───▶ psum[1]
              │1,0  │1,1  │1,2  │1,3  │     │1,15 │
              └──┬──┴──┬──┴──┬──┴──┬──┴─────┴──┬──┘
    ┌────┐       ↓     ↓     ↓     ↓           ↓
    │w[2]│───▶┌─────┬─────┬─────┬─────┬─────┬─────┐
    └────┘    │PE   │PE   │PE   │PE   │ ... │PE   │───▶ psum[2]
              │2,0  │2,1  │2,2  │2,3  │     │2,15 │
              └─────┴─────┴─────┴─────┴─────┴─────┘
                 :     :     :     :           :
    ┌────┐       ↓     ↓     ↓     ↓           ↓
    │w[15]│──▶┌─────┬─────┬─────┬─────┬─────┬─────┐
    └────┘    │PE   │PE   │PE   │PE   │ ... │PE   │───▶ psum[15]
              │15,0 │15,1 │15,2 │15,3 │     │15,15│
              └─────┴─────┴─────┴─────┴─────┴─────┘

    Weight      Each PE:                      Partial sums
    rows        • Stores 1 weight (INT8)      accumulate
    (16)        • MAC: acc += w × a           horizontally
                • Passes activation down      (INT32)
```

### Dataflow: Weight-Stationary

1. **Load Phase**: Weights loaded into PEs (stay fixed for entire tile)
2. **Compute Phase**: Activations stream through, MACs accumulate
3. **Drain Phase**: Partial sums collected from right edge

```
Cycle:    1    2    3    4    5    ...   K+15
         ┌────────────────────────────────────┐
Row 0:   │ a0   a1   a2   a3   ...   aK-1     │ → psum[0] complete
Row 1:   │      a0   a1   a2   ...   aK-2     │ → psum[1] complete  
Row 2:   │           a0   a1   ...   aK-3     │ → psum[2] complete
  :      │                                    │
Row 15:  │                          a0   ...  │ → psum[15] complete
         └────────────────────────────────────┘
          ◄──── K cycles + 15 skew cycles ────►
```

---

## 🧮 Processing Element (PE) Architecture

```
                    ┌─────────────────────────────────┐
                    │           PE [row, col]         │
    activation_in ─▶│  ┌─────┐                        │
    (INT8)          │  │ REG │─┬──────────────────────┼──▶ activation_out
                    │  └─────┘ │                      │     (to PE below)
                    │          ↓                      │
                    │     ┌─────────┐                 │
    weight_in ─────▶│────▶│   ×     │ INT8 × INT8     │
    (INT8)          │     │  (MUL)  │ = INT16         │
                    │     └────┬────┘                 │
                    │          ↓                      │
                    │     ┌─────────┐   ┌─────────┐   │
    psum_in ───────▶│────▶│    +    │──▶│   REG   │───┼──▶ psum_out
    (INT32)         │     │  (ACC)  │   │ (INT32) │   │    (to PE right)
                    │     └─────────┘   └─────────┘   │
                    │                                 │
                    └─────────────────────────────────┘

    Timing: 1 cycle latency (fully pipelined)
    Power:  ~0.5 mW per PE @ 200 MHz (estimated)
```

---

## 📦 BSR (Block Sparse Row) Format

The accelerator skips zero blocks entirely, saving compute and memory bandwidth.

### Memory Layout

```
Dense Matrix (64×64, ~70% block-sparse):        BSR Format:
┌────┬────┬────┬────┐                           
│████│    │████│    │  Block Row 0              Header (12 bytes):
│████│    │████│    │  (2 non-zero blocks)      ┌──────────────────┐
├────┼────┼────┼────┤                           │ nnz_blocks: 5    │ uint32
│    │████│    │    │  Block Row 1              │ num_blk_rows: 4  │ uint32
│    │████│    │    │  (1 non-zero block)       │ num_blk_cols: 4  │ uint32
├────┼────┼────┼────┤                           └──────────────────┘
│████│    │    │████│  Block Row 2              
│████│    │    │████│  (2 non-zero blocks)      row_ptr[5] (10 bytes):
├────┼────┼────┼────┤                           ┌─────────────────────────────┐
│    │    │    │    │  Block Row 3              │ 0 │ 2 │ 3 │ 5 │ 5 │        │
│    │    │    │    │  (0 non-zero blocks)      └─────────────────────────────┘
└────┴────┴────┴────┘                            ↑   ↑   ↑   ↑   ↑
                                                 │   │   │   │   └─ end (row 3)
Each block: 16×16 = 256 INT8 values              │   │   │   └─ start row 3
                                                 │   │   └─ start row 2
                                                 │   └─ start row 1
                                                 └─ start row 0 (always 0)

                                                col_idx[5] (10 bytes):
                                                ┌───────────────────────────┐
                                                │ 0 │ 2 │ 1 │ 0 │ 3 │       │
                                                └───────────────────────────┘
                                                  ↑   ↑   ↑   ↑   ↑
                                                  │   │   │   └───┴─ row 2 blocks
                                                  │   │   └─ row 1 block
                                                  └───┴─ row 0 blocks

                                                data[5 × 256] (1280 bytes):
                                                ┌─────────────────────────────┐
                                                │ Block(0,0) │ Block(0,2) │  │
                                                │ Block(1,1) │ Block(2,0) │  │
                                                │ Block(2,3) │            │  │
                                                └─────────────────────────────┘
```

### Sparsity Savings

| Sparsity | Dense Blocks | NNZ Blocks | Compute Savings |
|----------|-------------|------------|-----------------|
| 0%       | 16          | 16         | 0%              |
| 50%      | 16          | 8          | 50%             |
| 70%      | 16          | 5          | 69%             |
| 90%      | 16          | 2          | 88%             |

---

## 🔌 AXI Interface Connections

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ACCEL_TOP                                     │
│                                                                         │
│   AXI4-Lite Slave (Control)              AXI4 Master (Data)             │
│   ┌─────────────────────────┐            ┌─────────────────────────┐   │
│   │ Address   Register      │            │ Channel    Purpose      │   │
│   ├─────────────────────────┤            ├─────────────────────────┤   │
│   │ 0x00      CTRL          │            │ AR/R       Read weights │   │
│   │ 0x04      STATUS        │            │            Read acts    │   │
│   │ 0x08      BSR_ADDR_LO   │            │                         │   │
│   │ 0x0C      BSR_ADDR_HI   │            │ AW/W       Write output │   │
│   │ 0x10      ACT_ADDR_LO   │            │                         │   │
│   │ 0x14      ACT_ADDR_HI   │            │ Burst      Up to 256B   │   │
│   │ 0x18      OUT_ADDR_LO   │            │ Width      64-bit       │   │
│   │ 0x1C      OUT_ADDR_HI   │            └─────────────────────────┘   │
│   │ 0x20      TILE_CONFIG   │                                          │
│   │ 0x24      IRQ_ENABLE    │            AXI Stream (optional debug)   │
│   │ 0x28      IRQ_STATUS    │            ┌─────────────────────────┐   │
│   │ 0x2C      PERF_CYCLES   │            │ TDATA      256 bits     │   │
│   │ 0x30      PERF_STALLS   │            │ TVALID/TREADY           │   │
│   └─────────────────────────┘            └─────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### CSR Register Map

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| 0x00 | CTRL | RW | `[0]` START, `[1]` RESET, `[2]` IRQ_EN |
| 0x04 | STATUS | RO | `[0]` BUSY, `[1]` DONE, `[2]` ERROR |
| 0x08 | BSR_ADDR | RW | DDR address of BSR weight data |
| 0x10 | ACT_ADDR | RW | DDR address of activation data |
| 0x18 | OUT_ADDR | RW | DDR address for output results |
| 0x20 | TILE_CFG | RW | `[15:0]` M, `[31:16]` N, `[47:32]` K |
| 0x2C | CYCLES | RO | Performance counter: total cycles |
| 0x30 | STALLS | RO | Performance counter: stall cycles |

---

## 🎯 Zynq Z2 (PYNQ-Z2) Deployment

### Target: Xilinx XC7Z020-1CLG400C

#### Resource Utilization Estimates

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUTs     | ~18K | 53,200    | 34%         |
| FFs      | ~12K | 106,400   | 11%         |
| BRAM     | 64   | 140       | 46%         |
| DSP48    | 256  | 220       | **117%** ⚠️ |

> ⚠️ **Note**: 16×16 = 256 MACs exceeds Z7020's 220 DSPs. Options:
> 1. Use 14×14 array (196 DSPs, fits)
> 2. Use LUT-based multipliers for 36 PEs
> 3. Time-multiplex (2 cycles per MAC)

#### Recommended: 14×14 Array for Z7020

```
parameter N_ROWS = 14;  // Instead of 16
parameter N_COLS = 14;  // Fits in 196 DSPs
```

### Vivado Project Setup

```bash
# 1. Create project
vivado -mode batch -source scripts/create_project.tcl

# 2. Or manually:
cd hw/rtl
vivado &

# In Vivado GUI:
# - Create Project → RTL Project
# - Add sources: rtl/**/*.sv
# - Add constraints: constraints/pynq_z2.xdc
# - Set top: accel_top
```

### Block Design (Recommended)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Vivado Block Design                          │
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐     ┌──────────────┐   │
│  │   ZYNQ PS    │      │  AXI Inter-  │     │  accel_top   │   │
│  │              │      │  connect     │     │  (Your IP)   │   │
│  │  M_AXI_HPM0 ─┼─────▶│              │────▶│  S_AXI_LITE  │   │
│  │              │      │              │     │              │   │
│  │  S_AXI_HP0 ◀─┼──────│              │◀────│  M_AXI       │   │
│  │              │      └──────────────┘     │              │   │
│  │  FCLK_CLK0  ─┼──────────────────────────▶│  clk         │   │
│  │  (100 MHz)   │                           │              │   │
│  └──────────────┘                           └──────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Pin Constraints (pynq_z2.xdc)

```tcl
# Clock (directly from PS, no external pin needed)
# LEDs for debug
set_property PACKAGE_PIN R14 [get_ports {debug_led[0]}]
set_property PACKAGE_PIN P14 [get_ports {debug_led[1]}]
set_property PACKAGE_PIN N16 [get_ports {debug_led[2]}]
set_property PACKAGE_PIN M14 [get_ports {debug_led[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {debug_led[*]}]

# Buttons for manual control (optional)
set_property PACKAGE_PIN D19 [get_ports btn0]
set_property PACKAGE_PIN D20 [get_ports btn1]
set_property IOSTANDARD LVCMOS33 [get_ports btn*]
```

### PYNQ Python Driver

```python
from pynq import Overlay, allocate
import numpy as np

class AccelDriver:
    """PYNQ driver for ACCEL-v1 sparse accelerator."""
    
    # CSR offsets
    CTRL = 0x00
    STATUS = 0x04
    BSR_ADDR = 0x08
    ACT_ADDR = 0x10
    OUT_ADDR = 0x18
    TILE_CFG = 0x20
    
    def __init__(self, bitstream="accel_top.bit"):
        self.ol = Overlay(bitstream)
        self.accel = self.ol.accel_top_0
        self.dma = self.ol.axi_dma_0
        
    def run_gemm(self, weights_bsr: bytes, activations: np.ndarray) -> np.ndarray:
        """Run sparse GEMM on hardware."""
        M, K = activations.shape
        # ... allocate buffers, configure CSRs, start, wait ...
        
    def wait_done(self, timeout_ms=1000):
        """Poll STATUS register until DONE bit set."""
        import time
        start = time.time()
        while (time.time() - start) * 1000 < timeout_ms:
            status = self.accel.read(self.STATUS)
            if status & 0x2:  # DONE bit
                return True
            time.sleep(0.001)
        raise TimeoutError("Accelerator timeout")
```

### Build and Deploy

```bash
# 1. Synthesize and implement
cd vivado_project
vivado -mode batch -source build.tcl

# 2. Generate bitstream
# (Done by build.tcl, or: Flow → Generate Bitstream)

# 3. Copy to PYNQ board
scp accel_top.bit xilinx@pynq:/home/xilinx/

# 4. On PYNQ board
python3
>>> from accel_driver import AccelDriver
>>> accel = AccelDriver("accel_top.bit")
>>> accel.run_gemm(weights, activations)
```

---

## 📁 RTL File Structure

```
hw/rtl/
├── top/
│   ├── accel_top.sv           # Top-level with AXI interfaces
│   └── accel_top_dual_clk.sv  # Optional dual-clock version
├── systolic/
│   ├── systolic_array.sv      # 16×16 PE array
│   └── pe.sv                  # Single processing element
├── mac/
│   └── mac8.sv                # INT8 MAC unit
├── buffer/
│   ├── act_buffer.sv          # Double-buffered activations
│   ├── wgt_buffer.sv          # Weight BRAM
│   └── output_accumulator.sv  # Output collection + ReLU
├── dma/
│   ├── bsr_dma.sv             # BSR weight loader (AXI master)
│   └── act_dma.sv             # Activation streamer
├── control/
│   ├── csr.sv                 # Control/Status registers
│   ├── scheduler.sv           # Dense tile scheduler
│   └── bsr_scheduler.sv       # Sparse block scheduler
└── host_iface/
    ├── axi_lite_slave.sv      # AXI-Lite for CSRs
    └── axi_dma_bridge.sv      # AXI4 master wrapper
```

---

## 🧪 Simulation

### Verilator (Fast)

```bash
cd hw/sim
make -f Makefile.verilator test_systolic_array
./build/Vsystolic_array
```

### Icarus Verilog

```bash
cd hw/sim/sv
iverilog -g2012 -o systolic_tb.vvp \
    systolic_tb.sv \
    ../../rtl/systolic/*.sv \
    ../../rtl/mac/*.sv
vvp systolic_tb.vvp
```

### Cocotb (Python testbench)

```bash
cd hw/sim/cocotb
make SIM=verilator
```

---

## 📊 Performance Estimates

### ResNet-18 Inference

| Layer | M | N | K | Blocks (Dense) | Blocks (70% Sparse) | Cycles |
|-------|---|---|---|----------------|---------------------|--------|
| conv1 | 64 | 3136 | 147 | 11,200 | 3,360 | 35K |
| layer1.0.conv1 | 64 | 3136 | 576 | 86,400 | 25,920 | 270K |
| layer2.0.conv1 | 128 | 784 | 576 | 43,200 | 12,960 | 135K |
| layer3.0.conv1 | 256 | 196 | 1152 | 21,600 | 6,480 | 67K |
| layer4.0.conv1 | 512 | 49 | 2304 | 10,800 | 3,240 | 34K |
| fc | 1000 | 1 | 512 | 2,000 | 600 | 6K |

**Total: ~2.1M cycles @ 200 MHz = 10.5 ms/image (70% sparse)**

---

## 🔧 Customization

### Changing Array Size

Edit `rtl/systolic/systolic_array.sv`:
```systemverilog
module systolic_array #(
    parameter N_ROWS = 16,  // Change to 14 for Z7020
    parameter N_COLS = 16,  // Change to 14 for Z7020
    // ...
)
```

### Changing Block Size

Edit `rtl/control/bsr_scheduler.sv`:
```systemverilog
localparam BLOCK_SIZE = 16;  // Must match N_ROWS/N_COLS
```

And update C++ packer:
```cpp
// hw/sim/cpp/include/bsr_packer.hpp
constexpr std::size_t BSR_BLOCK_SIZE = 16;  // Keep in sync
```
