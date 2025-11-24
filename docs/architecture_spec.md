# ACCEL-v1 Architecture Specification
**8×8 Sparse Systolic Array Accelerator for PYNQ-Z2**

Version: 1.0
Date: November 24, 2025
Target: Zynq-7020 (PYNQ-Z2)

---

## 1. System Overview

ACCEL-v1 is a hardware accelerator for sparse matrix multiplication using Block Sparse Row (BSR) format. It implements an 8×8 systolic array with weight-stationary dataflow, optimized for CNN inference workloads like ResNet-50.

**Key Features:**
- 8×8 systolic array (64 PEs)
- INT8 quantization
- BSR sparse format support (90% sparsity)
- AXI4 interfaces for PS-PL communication
- 6.4 GOPS dense, 64 effective GOPS sparse

---

## 2. Data Path Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ARM PS (PYNQ Python)                              │
│  ┌──────────────┐                                                        │
│  │ Control CSRs │◄──────────── AXI4-Lite (GP0) ─────────────┐          │
│  └──────────────┘                                             │          │
│  ┌──────────────────────────────────────────────┐            │          │
│  │           DDR3 Memory (512 MB)               │            │          │
│  │  • BSR Metadata (row_ptr, col_idx)          │            │          │
│  │  • Block Data (8×8 INT8 blocks)             │            │          │
│  │  • Activation Vectors                        │            │          │
│  │  • Output Results                            │            │          │
│  └───────┬──────────────────────────────────────┘            │          │
│          │ ▲                                                  │          │
└──────────┼─┼──────────────────────────────────────────────────┼──────────┘
           │ │                                                  │
         HP0│ │HP0          AXI4 Master (64-bit, burst)        │AXI4-Lite
           │ │                                                  │(32-bit)
┌──────────▼─┴──────────────────────────────────────────────────┴──────────┐
│                         FPGA PL (Accelerator)                             │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │                    AXI DMA Bridge                                  │   │
│  │  • Burst reads from DDR (metadata + blocks)                       │   │
│  │  • Burst writes to DDR (results)                                  │   │
│  │  • Internal FIFOs (512 deep for data, 64 for metadata)           │   │
│  └─────┬─────────────────────────────────────────────────┬───────────┘   │
│        │                                                  │               │
│        │ metadata stream                                 │ block data    │
│        │ (row_ptr, col_idx)                             │ stream        │
│        ▼                                                  ▼               │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              Sparse Controller FSM                                │   │
│  │  • Parse BSR format (row_ptr → col_idx lookup)                   │   │
│  │  • Generate PE addresses for non-zero blocks                     │   │
│  │  • Skip zero blocks (sparse acceleration!)                       │   │
│  │  • Control weight loading & accumulation                         │   │
│  └─────┬────────────────────────────────────────────────────────────┘   │
│        │                                                                  │
│        │ weight_data[7:0], activation[7:0]                               │
│        │ pe_enable, load_weight, acc_clear                               │
│        ▼                                                                  │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                  8×8 Systolic Array                               │   │
│  │                                                                    │   │
│  │   PE00──PE01──PE02──PE03──PE04──PE05──PE06──PE07                │   │
│  │    │    │    │    │    │    │    │    │                          │   │
│  │   PE10──PE11──PE12──PE13──PE14──PE15──PE16──PE17                │   │
│  │    │    │    │    │    │    │    │    │                          │   │
│  │   PE20──PE21──PE22──PE23──PE24──PE25──PE26──PE27                │   │
│  │    │    │    │    │    │    │    │    │                          │   │
│  │   PE30──PE31──PE32──PE33──PE34──PE35──PE36──PE37                │   │
│  │    │    │    │    │    │    │    │    │                          │   │
│  │   PE40──PE41──PE42──PE43──PE44──PE45──PE46──PE47                │   │
│  │    │    │    │    │    │    │    │    │                          │   │
│  │   PE50──PE51──PE52──PE53──PE54──PE55──PE56──PE57                │   │
│  │    │    │    │    │    │    │    │    │                          │   │
│  │   PE60──PE61──PE62──PE63──PE64──PE65──PE66──PE67                │   │
│  │    │    │    │    │    │    │    │    │                          │   │
│  │   PE70──PE71──PE72──PE73──PE74──PE75──PE76──PE77                │   │
│  │    │    │    │    │    │    │    │    │                          │   │
│  │    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼                          │   │
│  │  [Accumulator outputs: 32-bit results]                           │   │
│  │                                                                    │   │
│  │  Dataflow:                                                        │   │
│  │   • Weights: Broadcast vertically, held stationary in each PE    │   │
│  │   • Activations: Flow horizontally left→right                    │   │
│  │   • Partials: Accumulate locally in each PE                      │   │
│  └─────┬──────────────────────────────────────────────────────────┘   │
│        │                                                                │
│        │ result_valid, result_data[31:0]                               │
│        ▼                                                                │
│  ┌────────────────────────────────────────┐                           │
│  │   Output Buffer (FIFO, 64 deep)       │                           │
│  └────────────┬───────────────────────────┘                           │
│               │                                                         │
│               └──► Back to AXI DMA Bridge (write to DDR)              │
│                                                                         │
│  ┌────────────────────────────────────────┐                           │
│  │   CSR Slave (Control/Status Regs)     │◄─────AXI4-Lite            │
│  │   • START, DONE, ERROR registers       │                           │
│  │   • Performance counters               │                           │
│  └────────────────────────────────────────┘                           │
└───────────────────────────────────────────────────────────────────────┘

Clock: 100 MHz
```

---

## 3. Module Hierarchy

```
accel_top
├── axi_lite_slave (CSR interface)
│   └── Registers: CTRL, STATUS, PERF_CYCLES, etc.
│
├── axi_dma_bridge (Memory interface)
│   ├── axi_master_read (64-bit bursts from DDR)
│   ├── axi_master_write (64-bit bursts to DDR)
│   ├── metadata_fifo (512 deep, 32-bit wide)
│   └── data_fifo (512 deep, 64-bit wide)
│
├── sparse_controller (BSR parser & scheduler)
│   ├── bsr_parser (reads row_ptr, col_idx)
│   ├── block_scheduler (generates PE coordinates)
│   └── weight_loader (loads 8×8 blocks into array)
│
├── systolic_array_8x8 (Compute core)
│   ├── pe_array [8][8] (64 PEs total)
│   │   └── pe (MAC unit with weight register)
│   └── output_collector (gathers results)
│
└── output_buffer (Result FIFO, 64 deep)
```

---

## 4. Interface Specifications

### 4.1 External Interfaces (PS ↔ PL)

#### AXI4-Lite Slave (Control/Status via GP0)

| Port Name       | Direction | Width | Protocol  | Purpose         |
|-----------------|-----------|-------|-----------|-----------------|
| s_axi_awaddr    | Input     | 12    | AXI4-Lite | Write address   |
| s_axi_awvalid   | Input     | 1     |           | Address valid   |
| s_axi_awready   | Output    | 1     |           | Address ready   |
| s_axi_wdata     | Input     | 32    |           | Write data      |
| s_axi_wvalid    | Input     | 1     |           | Data valid      |
| s_axi_wready    | Output    | 1     |           | Data ready      |
| s_axi_bresp     | Output    | 2     |           | Write response  |
| s_axi_bvalid    | Output    | 1     |           | Response valid  |
| s_axi_bready    | Input     | 1     |           | Response ready  |
| s_axi_araddr    | Input     | 12    |           | Read address    |
| s_axi_arvalid   | Input     | 1     |           | Address valid   |
| s_axi_arready   | Output    | 1     |           | Address ready   |
| s_axi_rdata     | Output    | 32    |           | Read data       |
| s_axi_rresp     | Output    | 2     |           | Read response   |
| s_axi_rvalid    | Output    | 1     |           | Data valid      |
| s_axi_rready    | Input     | 1     |           | Data ready      |

**CSR Register Map (32-bit registers):**

| Offset | Name              | Description                        | Access |
|--------|-------------------|------------------------------------|--------|
| 0x00   | CTRL              | [0]: START, [1]: RESET            | RW     |
| 0x04   | STATUS            | [0]: DONE, [1]: BUSY, [2]: ERROR  | RO     |
| 0x08   | ROW_PTR_ADDR_LO   | DDR address for row_ptr (lower)   | RW     |
| 0x0C   | ROW_PTR_ADDR_HI   | DDR address for row_ptr (upper)   | RW     |
| 0x10   | COL_IDX_ADDR_LO   | DDR address for col_idx (lower)   | RW     |
| 0x14   | COL_IDX_ADDR_HI   | DDR address for col_idx (upper)   | RW     |
| 0x18   | BLOCK_DATA_ADDR_LO| DDR address for block_data (lower)| RW     |
| 0x1C   | BLOCK_DATA_ADDR_HI| DDR address for block_data (upper)| RW     |
| 0x20   | OUTPUT_ADDR_LO    | DDR address for results (lower)   | RW     |
| 0x24   | OUTPUT_ADDR_HI    | DDR address for results (upper)   | RW     |
| 0x28   | NUM_ROWS          | Matrix number of rows             | RW     |
| 0x2C   | NUM_COLS          | Matrix number of columns          | RW     |
| 0x30   | PERF_CYCLES       | Total cycles elapsed              | RO     |
| 0x34   | PERF_BLOCKS       | Blocks processed                  | RO     |

#### AXI4 Master (Data Movement via HP0)

| Port Name       | Direction | Width | Protocol | Purpose          |
|-----------------|-----------|-------|----------|------------------|
| m_axi_awaddr    | Output    | 32    | AXI4     | Write address    |
| m_axi_awlen     | Output    | 8     |          | Burst length-1   |
| m_axi_awsize    | Output    | 3     |          | Burst size       |
| m_axi_awburst   | Output    | 2     |          | Burst type       |
| m_axi_awvalid   | Output    | 1     |          | Address valid    |
| m_axi_awready   | Input     | 1     |          | Address ready    |
| m_axi_wdata     | Output    | 64    |          | Write data       |
| m_axi_wstrb     | Output    | 8     |          | Write strobes    |
| m_axi_wlast     | Output    | 1     |          | Last beat        |
| m_axi_wvalid    | Output    | 1     |          | Data valid       |
| m_axi_wready    | Input     | 1     |          | Data ready       |
| m_axi_bresp     | Input     | 2     |          | Write response   |
| m_axi_bvalid    | Input     | 1     |          | Response valid   |
| m_axi_bready    | Output    | 1     |          | Response ready   |
| m_axi_araddr    | Output    | 32    |          | Read address     |
| m_axi_arlen     | Output    | 8     |          | Burst length-1   |
| m_axi_arsize    | Output    | 3     |          | Burst size       |
| m_axi_arburst   | Output    | 2     |          | Burst type       |
| m_axi_arvalid   | Output    | 1     |          | Address valid    |
| m_axi_arready   | Input     | 1     |          | Address ready    |
| m_axi_rdata     | Input     | 64    |          | Read data        |
| m_axi_rresp     | Input     | 2     |          | Read response    |
| m_axi_rlast     | Input     | 1     |          | Last beat        |
| m_axi_rvalid    | Input     | 1     |          | Data valid       |
| m_axi_rready    | Output    | 1     |          | Data ready       |

**Configuration:**
- Data Width: 64-bit
- Burst Length: Up to 16 beats
- Burst Type: INCR (incrementing)

### 4.2 Internal Interfaces

#### Sparse Controller → Systolic Array

| Signal Name      | Width | Direction | Purpose                  |
|------------------|-------|-----------|--------------------------|
| weight_data      | 8     | Output    | Weight value to load     |
| activation_data  | 8     | Output    | Activation input         |
| pe_row_sel       | 3     | Output    | Target PE row (0-7)      |
| pe_col_sel       | 3     | Output    | Target PE column (0-7)   |
| load_weight      | 1     | Output    | Load weight into PE      |
| acc_clear        | 1     | Output    | Clear accumulator        |
| enable           | 1     | Output    | Enable MAC operation     |
| result_ready     | 1     | Input     | Result available         |
| result_data      | 32    | Input     | MAC result output        |
| result_row       | 3     | Input     | Result PE row            |
| result_col       | 3     | Input     | Result PE column         |

#### Systolic Array Internal (PE-to-PE)

| Signal Name | Width | Direction    | Purpose                        |
|-------------|-------|--------------|--------------------------------|
| a_in        | 8     | Input        | Activation input (horizontal)  |
| a_out       | 8     | Output       | Activation output (to next PE) |
| b_in        | 8     | Input        | Weight input (broadcast)       |
| load_weight | 1     | Input        | Load weight signal             |
| enable      | 1     | Input        | Enable MAC                     |
| acc_clear   | 1     | Input        | Clear accumulator              |
| acc_out     | 32    | Output       | Accumulated result             |

---

## 5. Performance Analysis

### 5.1 Theoretical Peak GOPS

**Dense Computation:**
```
Array size: 8 × 8 = 64 PEs
Operations: 1 MAC per PE per cycle
Clock: 100 MHz

Peak GOPS = 64 PEs × 1 MAC/cycle × 100 MHz
          = 6,400 MMAC/s
          = 6.4 GOPS
```

### 5.2 Sparse Efficiency Model

**With 90% Sparsity (BSR 8×8 blocks):**

Assuming a matrix with 100 dense blocks becomes 10 non-zero blocks after sparsification:

```
Dense time:  100 blocks × 64 cycles/block = 6,400 cycles
Sparse time: 10 blocks × 76 cycles/block = 760 cycles

Theoretical speedup = 6,400 / 760 = 8.4×

Where 76 cycles/block = 64 (compute) + 10 (metadata fetch) + 2 (control)
Compute efficiency = 64/76 = 84%
```

**Effective Sparse GOPS:**
```
Effective GOPS = 6.4 GOPS × 8.4 = 53.8 GOPS
```

For 90% sparsity, theoretical speedup is 10×, but overhead reduces it to ~8.4× in practice.

### 5.3 Bandwidth Requirements

#### Per 8×8 Block Computation:

**Input Bandwidth (Read from DDR):**
```
Metadata:
  • row_ptr: 4 bytes
  • col_idx: 4 bytes
  Total: 8 bytes/block

Data:
  • 8×8 INT8 weights: 64 bytes
  • 8×1 INT8 activations: 8 bytes
  Total: 72 bytes/block

Total read per block = 80 bytes
```

**Output Bandwidth (Write to DDR):**
```
Results:
  • 8×8 INT32 outputs: 64 × 4 = 256 bytes/block
```

#### Throughput Analysis:

```
Block computation time = 76 cycles @ 100 MHz = 760 ns
Blocks per second = 1.32 million blocks/sec

Read bandwidth  = 1.32M × 80 bytes  = 106 MB/s
Write bandwidth = 1.32M × 256 bytes = 337 MB/s
Total bandwidth = 443 MB/s

HP0 port capability: ~1200 MB/s @ 150 MHz
Utilization: 443 / 1200 = 37% ✓
```

**Conclusion:** Bandwidth is NOT the bottleneck. Computation is the limiting factor.

---

## 6. Resource Utilization (Estimated)

**Target: Zynq-7020 (xc7z020clg400-1)**

| Resource | Available | Used (Est.) | Utilization |
|----------|-----------|-------------|-------------|
| LUTs     | 53,200    | ~15,000     | 28%         |
| FFs      | 106,400   | ~20,000     | 19%         |
| DSPs     | 220       | 64          | 29%         |
| BRAM     | 140       | ~20         | 14%         |

**Breakdown by Module:**
- Systolic Array (64 PEs): 64 DSPs, ~8,000 LUTs
- AXI DMA Bridge: ~3,000 LUTs, 10 BRAMs
- Sparse Controller: ~2,000 LUTs
- CSR/Control: ~1,000 LUTs

---

## 7. Timing Constraints

**Clock Domains:**
- Main clock: 100 MHz (10 ns period)
- AXI clock: 100 MHz (same as main)

**Critical Paths:**
- PE MAC operation: ~8 ns
- AXI read/write: ~9 ns
- FSM state transitions: ~5 ns

**Timing Margin:** ~1-2 ns (meets 100 MHz timing)

---

## 8. Power Estimation

**Approximate Power Breakdown @ 100 MHz:**

| Component        | Power (mW) |
|------------------|------------|
| Systolic Array   | 800        |
| AXI Interface    | 400        |
| Memory (BRAM)    | 200        |
| Control Logic    | 200        |
| Clock Network    | 200        |
| **Total**        | **1,800**  |

**Energy per Operation:**
```
Energy per MAC = 1.8W / (6.4 GOPS) = 0.28 nJ/MAC

With 90% sparsity:
Effective energy = 0.28 nJ / 8.4 = 0.033 nJ/MAC
```

---

## 9. Dataflow Details

### Weight-Stationary Dataflow

**Weight Loading Phase:**
1. Broadcast weight to entire column of PEs
2. Each PE captures weight in local register
3. Weight remains stationary during computation

**Computation Phase:**
1. Activations flow horizontally left→right
2. Each PE multiplies stationary weight × flowing activation
3. Partial sums accumulate locally in each PE
4. No partial sum movement during computation

**Result Collection:**
1. After computation completes, read accumulated results
2. Results move only once (from PE to output buffer)

**Energy Benefits:**
- Weights loaded once, used N times (where N = activation vector length)
- Activations move only horizontally (single hop per PE)
- Partial sums don't move until final readout
- Minimal data movement = minimal energy

---

## 10. Sparse Format: BSR (Block Sparse Row)

### Memory Layout (Hardware-Aligned Packing)

**Three arrays stored in DDR:**

1. **row_ptr[num_rows + 1]:**
   - Each entry: 4 bytes (32-bit word)
   - Lower 16 bits: index into col_idx/block_data
   - Address increment: 4 bytes

2. **col_idx[num_nonzero_blocks]:**
   - Each entry: 4 bytes (32-bit word)
   - Lower 16 bits: column index of block
   - Address increment: 4 bytes

3. **block_data[num_nonzero_blocks][64]:**
   - Each block: 64 INT8 values = 64 bytes
   - Stored sequentially

**Example for MNIST FC1 (128×784, 90% sparse):**
```
row_ptr:     129 entries × 4 bytes = 516 bytes
col_idx:     ~100 entries × 4 bytes = 400 bytes
block_data:  ~100 blocks × 64 bytes = 6,400 bytes
Total: ~7.3 KB
```

---

## 11. Verification Plan

### Unit Tests
1. **PE (Processing Element)**
   - Test MAC operation with known inputs
   - Test weight loading and holding
   - Test accumulator clear

2. **Systolic Array 8×8**
   - Test weight broadcast to columns
   - Test activation propagation
   - Test result collection

3. **BSR Parser**
   - Test row_ptr indexing
   - Test col_idx lookup
   - Test block address calculation

4. **AXI DMA Bridge**
   - Test burst reads
   - Test burst writes
   - Test FIFO overflow/underflow

### Integration Tests
1. Small sparse matrix (4×4, 2 non-zero blocks)
2. MNIST FC1 layer (128×784)
3. ResNet-50 first layer

### System Tests
1. End-to-end inference on PYNQ
2. Performance measurement (cycles, GOPS)
3. Accuracy verification (compare with PyTorch)

---

## 12. References

**Papers:**
- "In-Datacenter Performance Analysis of a Tensor Processing Unit" (Google TPU)
- "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep CNNs" (MIT)
- "EIE: Efficient Inference Engine on Compressed Deep Neural Networks" (Stanford)

**Datasheets:**
- Zynq-7000 SoC Technical Reference Manual (UG585)
- PYNQ-Z2 Board Reference Manual

**Tools:**
- Xilinx Vivado 2023.2
- PYNQ v3.0.1
- Python 3.10 + NumPy/PyTorch
