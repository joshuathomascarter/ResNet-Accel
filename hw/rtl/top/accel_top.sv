// =============================================================================
// accel_top.sv — AXI4-based Sparse INT8 Systolic Array Accelerator Top-Level
// =============================================================================
//
// ARCHITECTURE OVERVIEW:
// ----------------------
// This is the top-level integration module for a sparse INT8 matrix multiply
// accelerator targeting Zynq-7020 (PYNQ-Z2). It implements a weight-stationary
// dataflow with Block Sparse Row (BSR) compression for weight matrices.
//
//   ┌─────────────────────────────────────────────────────────────────────┐
//   │                         Zynq PS (ARM Cortex-A9)                     │
//   │                                                                     │
//   │   Linux Driver                                                      │
//   │       │                                                             │
//   │       ├── AXI-Lite GP0 ─────────────────┐                          │
//   │       │   (CSR R/W)                     │                          │
//   │       │                                 │                          │
//   │       └── mmap DDR ──────────────┐      │                          │
//   │           (Activations, Weights) │      │                          │
//   └───────────────────────────────────┼──────┼──────────────────────────┘
//                                       │      │
//   ┌───────────────────────────────────┼──────┼──────────────────────────┐
//   │                            PL (FPGA Fabric)                         │
//   │                                   │      │                          │
//   │   ┌───────────────────────────────┼──────▼──────┐                  │
//   │   │            AXI Interconnect                 │                  │
//   │   └───────────┬───────────────────┬─────────────┘                  │
//   │               │                   │                                 │
//   │     AXI4 HP0/HP2                 AXI-Lite                          │
//   │     (64-bit DDR)                 (32-bit CSR)                      │
//   │               │                   │                                 │
//   │   ┌───────────▼───────────┐   ┌───▼────────────┐                   │
//   │   │   axi_dma_bridge      │   │ axi_lite_slave │                   │
//   │   │   (2:1 Arbiter)       │   └───────┬────────┘                   │
//   │   └───┬───────────────┬───┘           │                            │
//   │       │               │               │                            │
//   │   ┌───▼───┐       ┌───▼───┐       ┌───▼───┐                        │
//   │   │act_dma│       │bsr_dma│       │  csr  │                        │
//   │   └───┬───┘       └───┬───┘       └───────┘                        │
//   │       │               │                                             │
//   │   ┌───▼───┐       ┌───▼───────────────────┐                        │
//   │   │act_buf│       │ BSR Metadata BRAMs    │                        │
//   │   │ (RAM) │       │ (row_ptr, col_idx,    │                        │
//   │   └───┬───┘       │  wgt_block)           │                        │
//   │       │           └───┬───────────────────┘                        │
//   │       │               │                                             │
//   │       │           ┌───▼────────┐                                   │
//   │       │           │meta_decode │                                   │
//   │       │           │  (Cache)   │                                   │
//   │       │           └───┬────────┘                                   │
//   │       │               │                                             │
//   │       │         ┌─────▼──────┐                                     │
//   │       │         │bsr_scheduler│                                    │
//   │       │         └─────┬──────┘                                     │
//   │       │               │                                             │
//   │       │     ┌─────────▼─────────┐                                  │
//   │       └────►│ systolic_array    │                                  │
//   │             │ (14×14 PE Grid)   │                                  │
//   │             └─────────┬─────────┘                                  │
//   │                       │                                             │
//   │             ┌─────────▼─────────┐                                  │
//   │             │  Output / Accum   │─────► DMA back to DDR            │
//   │             └───────────────────┘       (future: write DMA)        │
//   │                                                                     │
//   │             ┌───────────────────┐                                  │
//   │             │       perf        │                                  │
//   │             │ (Cycle Counters)  │                                  │
//   │             └───────────────────┘                                  │
//   └─────────────────────────────────────────────────────────────────────┘
//
// DATA FLOW:
// ----------
// 1. Host writes configuration to CSR (matrix dims, DMA addresses)
// 2. Host triggers DMA start pulses:
//    - act_dma: Loads dense activations from DDR → act_buffer RAM
//    - bsr_dma: Loads BSR sparse weights from DDR → metadata BRAMs
// 3. Host triggers start_pulse to begin computation
// 4. bsr_scheduler iterates through non-zero blocks:
//    a. Fetches row_ptr[i] and row_ptr[i+1] to determine column range
//    b. For each column index, loads weight block and streams activations
//    c. Signals systolic array with load_weight, pe_en, accum_en
// 5. systolic_array_sparse performs 14×14 MAC operations
// 6. Results accumulate in output registers (visible via CSR)
// 7. done signal asserted when all blocks processed
//
// SPARSE FORMAT (BSR - Block Sparse Row):
// ---------------------------------------
// Memory layout for a matrix with M=28, K=28, block_size=14:
//   DDR Base + 0x00: num_rows (= M/14 = 2)
//   DDR Base + 0x04: num_cols (= K/14 = 2)
//   DDR Base + 0x08: nnz_blocks (number of non-zero 14×14 blocks)
//   DDR Base + 0x0C: block_size (= 14)
//   DDR Base + 0x10: row_ptr[0..num_rows] (32-bit each)
//   DDR Base + 0x10 + (num_rows+1)*4: col_idx[0..nnz_blocks-1] (16-bit each)
//   DDR Base + aligned: blocks[0..nnz_blocks-1] (196 bytes each)
//
// DESIGN CHOICES:
// ---------------
// - SPARSE-ONLY: Dense systolic array variant removed to save ~40% LUT area.
//   For dense layers (fc1, fc2), use 100% dense BSR (no zero blocks).
// - READ-ONLY DMAs: Outputs are read via CSR result registers. Write DMA
//   for large output tensors is a future enhancement.
// - BRAM Sizing: BRAM_ADDR_W=10 gives 1024 entries. With 14×14 blocks:
//   - row_ptr_bram: 1024 × 32-bit = 4 KB (supports up to 1023 block rows)
//   - col_idx_bram: 1024 × 16-bit = 2 KB (supports up to 1024 non-zero blocks)
//   - wgt_block_bram: 1024 × 64-bit = 8 KB (25 full blocks, but typically
//     only current block needs to be resident)
//
// TIMING CONSTRAINTS:
// -------------------
// Target: 100 MHz (10 ns period) on Zynq-7020
// Critical paths:
//   - AXI read data → BRAM write: 3 cycles (DMA FSM latency)
//   - Metadata BRAM read → scheduler: 2 cycles (read-after-write hazard)
//   - Scheduler → systolic array control: 1 cycle
//
// RESOURCE ESTIMATES (Zynq-7020):
// -------------------------------
//   Module            | LUTs  | FFs   | BRAM36 | DSPs
//   ------------------|-------|-------|--------|------
//   axi_lite_slave    |  200  |  150  |   0    |  0
//   csr               |  400  |  500  |   0    |  0
//   axi_dma_bridge    |  300  |  200  |   0    |  0
//   act_dma           |  250  |  180  |   0    |  0
//   bsr_dma           |  450  |  350  |   0    |  0
//   meta_decode       |  200  |  150  |   1    |  0
//   bsr_scheduler     |  350  |  280  |   0    |  0
//   systolic_sparse   | 4000  | 8000  |   0    | 196
//   BRAMs (inline)    |    0  |    0  |   4    |  0
//   perf              |  100  |  200  |   0    |  0
//   ------------------|-------|-------|--------|------
//   Total (approx)    | 6250  |10000  |   5    | 196
//
// Zynq-7020 has 53,200 LUTs, 106,400 FFs, 140 BRAM36, 220 DSPs.
// This design uses ~12% LUTs, ~10% FFs, ~4% BRAM, ~89% DSPs.
//
// VERIFICATION:
// -------------
// - Cocotb testbench: hw/sim/cocotb/test_accel_top.py
// - C++ testbench: hw/sim/cpp/accel_top_tb.cpp
// - Waveform: Run with +wave=accel_top.vcd
//
// INTEGRATION:
// ------------
// Vivado block design connections:
//   - clk: pl_clk0 (100 MHz from FCLK_CLK0)
//   - rst_n: peripheral_aresetn from proc_sys_reset
//   - m_axi_*: HP0 or HP2 port of Zynq PS
//   - s_axi_*: M_AXI_GP0 through AXI Interconnect
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module accel_top #(
    // =========================================================================
    // SYSTOLIC ARRAY PARAMETERS
    // =========================================================================
    // N_ROWS, N_COLS: Systolic array dimensions.
    // - 14×14 chosen to balance DSP utilization (196 DSPs) with routing
    // - Matches BSR block size for efficient weight loading
    // - Larger arrays (16×16=256 DSPs) exceed Zynq-7020's 220 DSP limit
    parameter N_ROWS     = 14,
    parameter N_COLS     = 14,
    
    // DATA_W: Input data width (INT8 for quantized inference)
    // - INT8 provides 2× throughput vs INT16, 4× vs FP32
    // - Sufficient precision for CNN inference (< 1% accuracy loss)
    parameter DATA_W     = 8,
    
    // ACC_W: Accumulator width to prevent overflow
    // - 8-bit × 8-bit = 16-bit product
    // - Sum of K products: need log2(K) + 16 bits
    // - For K=512 (ResNet-18 max), need 16 + 9 = 25 bits minimum
    // - 32 bits provides headroom for larger layers
    parameter ACC_W      = 32,
    
    // =========================================================================
    // AXI INTERFACE PARAMETERS
    // =========================================================================
    // AXI_ADDR_W: 32-bit addresses for Zynq DDR (512 MB - 1 GB range)
    parameter AXI_ADDR_W = 32,
    
    // AXI_DATA_W: 64-bit for HP port bandwidth
    // - Zynq HP ports are 64-bit native
    // - 8 INT8 values per beat × 16 beats = 128 bytes/burst
    parameter AXI_DATA_W = 64,
    
    // AXI_ID_W: Transaction ID for out-of-order completion tracking
    // - 4 bits = 16 outstanding transactions (more than needed)
    // - ID=0 for BSR DMA, ID=1 for Act DMA
    parameter AXI_ID_W   = 4,
    
    // =========================================================================
    // INTERNAL MEMORY PARAMETERS
    // =========================================================================
    // CSR_ADDR_W: 8-bit CSR address (256 registers × 4 bytes = 1 KB)
    // - Matches Vivado AXI peripheral standard address range
    parameter CSR_ADDR_W = 8,
    
    // BRAM_ADDR_W: 10-bit BRAM address (1024 entries)
    // - row_ptr: 1024 × 4B = 4 KB (up to 1023 block rows)
    // - col_idx: 1024 × 2B = 2 KB (up to 1024 non-zero blocks)
    // - wgt_block: 1024 × 8B = 8 KB (block data staging)
    parameter BRAM_ADDR_W = 10
)(
    // =========================================================================
    // Clock & Reset
    // =========================================================================
    // clk: Main system clock (100 MHz from Zynq FCLK_CLK0)
    // - All modules operate on single clock domain
    // - See accel_top_dual_clk.sv for CDC version with separate AXI clock
    input  wire clk,
    
    // rst_n: Active-low synchronous reset
    // - Directly from Zynq proc_sys_reset peripheral_aresetn
    // - Must be held low for at least 16 cycles after clk is stable
    input  wire rst_n,

    // =========================================================================
    // AXI4 Master Interface (Read-Only, To DDR via Zynq HP Port)
    // =========================================================================
    // This interface connects to Zynq HP0 or HP2 for high-bandwidth DDR access.
    // Write channels omitted as this design only reads from DDR.
    // Future: Add write channels for output DMA.
    //
    // Zynq HP Port Capabilities:
    // - 64-bit data width, 1066 MT/s DDR3 → ~4 GB/s per port
    // - Supports outstanding transactions for latency hiding
    // - QoS (Quality of Service) configurable in Vivado
    //
    // -------------------------------------------------------------------------
    // Read Address Channel
    // -------------------------------------------------------------------------
    // m_axi_arid: Transaction ID (ID=0 for BSR DMA, ID=1 for Act DMA)
    // - Allows out-of-order completion if needed (not currently used)
    output wire [AXI_ID_W-1:0]   m_axi_arid,
    
    // m_axi_araddr: Byte address in DDR space (0x0000_0000 - 0x3FFF_FFFF)
    // - Must be aligned to transfer size (8-byte for 64-bit)
    output wire [AXI_ADDR_W-1:0] m_axi_araddr,
    
    // m_axi_arlen: Burst length - 1 (0 = 1 beat, 15 = 16 beats)
    // - 16 beats × 8 bytes = 128 bytes per burst (optimal for DDR)
    output wire [7:0]            m_axi_arlen,
    
    // m_axi_arsize: Bytes per beat = 2^arsize (3'b011 = 8 bytes)
    output wire [2:0]            m_axi_arsize,
    
    // m_axi_arburst: Burst type (2'b01 = INCR, incrementing addresses)
    output wire [1:0]            m_axi_arburst,
    
    // m_axi_arvalid/arready: Standard AXI handshake
    output wire                  m_axi_arvalid,
    input  wire                  m_axi_arready,

    // -------------------------------------------------------------------------
    // Read Data Channel
    // -------------------------------------------------------------------------
    // m_axi_rid: Echoed transaction ID (matches arid for request tracking)
    input  wire [AXI_ID_W-1:0]   m_axi_rid,
    
    // m_axi_rdata: 64-bit read data from DDR
    // - Contains 8 INT8 values or metadata (row_ptr, col_idx)
    input  wire [AXI_DATA_W-1:0] m_axi_rdata,
    
    // m_axi_rresp: Read response (2'b00 = OKAY, 2'b10 = SLVERR)
    input  wire [1:0]            m_axi_rresp,
    
    // m_axi_rlast: Last beat of burst (asserted on final transfer)
    input  wire                  m_axi_rlast,
    
    // m_axi_rvalid/rready: Data channel handshake
    input  wire                  m_axi_rvalid,
    output wire                  m_axi_rready,

    // =========================================================================
    // AXI4-Lite Slave Interface (CSR from Zynq GP Port)
    // =========================================================================
    // This interface connects to Zynq M_AXI_GP0 through an AXI Interconnect.
    // Provides register access for configuration, control, and status.
    //
    // GP Port Characteristics:
    // - 32-bit data width
    // - Lower bandwidth than HP (suitable for control registers)
    // - Directly accessible from Linux via /dev/mem or UIO driver
    //
    // Address Map (byte addresses, CSR base typically 0x4000_0000):
    //   0x00: CTRL      - Start/abort control
    //   0x04: STATUS    - Busy/done/error flags
    //   0x10-0x1C: Matrix dimensions (M, N, K)
    //   0x40-0x4C: DMA addresses and lengths
    //   0x80-0xBC: Performance counters (read-only)
    //   0xC0-0xFC: Result data (first 16 accumulators)
    //
    // -------------------------------------------------------------------------
    // Write Address Channel
    input  wire [AXI_ADDR_W-1:0] s_axi_awaddr,
    input  wire [2:0]            s_axi_awprot,   // Protection (ignored)
    input  wire                  s_axi_awvalid,
    output wire                  s_axi_awready,
    
    // Write Data Channel
    input  wire [31:0]           s_axi_wdata,
    input  wire [3:0]            s_axi_wstrb,    // Byte strobes (all 1s for word writes)
    input  wire                  s_axi_wvalid,
    output wire                  s_axi_wready,
    
    // Write Response Channel
    output wire [1:0]            s_axi_bresp,    // Always OKAY
    output wire                  s_axi_bvalid,
    input  wire                  s_axi_bready,
    
    // Read Address Channel
    input  wire [AXI_ADDR_W-1:0] s_axi_araddr,
    input  wire [2:0]            s_axi_arprot,   // Protection (ignored)
    input  wire                  s_axi_arvalid,
    output wire                  s_axi_arready,
    
    // Read Data Channel
    output wire [31:0]           s_axi_rdata,
    output wire [1:0]            s_axi_rresp,    // Always OKAY
    output wire                  s_axi_rvalid,
    input  wire                  s_axi_rready,

    // =========================================================================
    // Status Outputs (directly from CSR control signals)
    // =========================================================================
    // These signals can be routed to LEDs or external pins for debugging.
    // Also readable via CSR STATUS register.
    //
    // busy: Accelerator is processing (act_dma_busy | bsr_dma_busy | sched_busy)
    output wire busy,
    
    // done: Computation complete (tile/layer finished)
    output wire done,
    
    // error: DMA error occurred (address fault, bus error)
    output wire error
);

    // =========================================================================
    // Internal Signal Declarations
    // =========================================================================
    // Signal naming convention:
    //   - cfg_*: Configuration values from CSR
    //   - *_dma_*: DMA control/status signals
    //   - *_we, *_waddr, *_wdata: BRAM write interface
    //   - *_rd_*, *_rdata: BRAM read interface
    //   - sched_*: Scheduler control signals
    //   - meta_*: Metadata decoder interface

    // -------------------------------------------------------------------------
    // CSR Interface Signals
    // -------------------------------------------------------------------------
    // CSR bus signals (from axi_lite_slave to csr module)
    wire                    csr_wen, csr_ren;
    wire [CSR_ADDR_W-1:0]   csr_addr;
    wire [31:0]             csr_wdata, csr_rdata;
    
    // Control pulses (W1P - Write-1-Pulse, self-clearing)
    wire                    start_pulse;    // Triggers computation start
    wire                    abort_pulse;    // Emergency stop (clears all state)

    // CSR Configuration Outputs (directly from register values)
    wire [31:0] cfg_M, cfg_N, cfg_K;        // Matrix dimensions
    wire [31:0] cfg_act_src_addr;           // Activation DMA source address
    wire [31:0] cfg_bsr_src_addr;           // BSR DMA source address
    wire [31:0] cfg_act_xfer_len;           // Activation transfer length (bytes)
    wire        cfg_act_dma_start;          // Activation DMA start pulse

    // -------------------------------------------------------------------------
    // DMA Control Signals
    // -------------------------------------------------------------------------
    // Activation DMA (loads dense activation vectors from DDR)
    // - Transfers contiguous data from cfg_act_src_addr
    // - Writes to act_buffer RAM
    wire                    act_dma_start;      // Start pulse (from CSR)
    wire                    act_dma_done;       // Transfer complete
    wire                    act_dma_busy;       // Transfer in progress
    wire                    act_dma_error;      // Bus error or timeout
    wire [AXI_ADDR_W-1:0]   act_dma_src_addr;   // DDR source address
    wire [31:0]             act_dma_xfer_len;   // Bytes to transfer

    // BSR DMA (loads sparse weight structure from DDR)
    // - Parses BSR header, row_ptr, col_idx, and weight blocks
    // - Writes to respective BRAM banks
    wire                    bsr_dma_start;      // Start pulse (from CSR)
    wire                    bsr_dma_done;       // All phases complete
    wire                    bsr_dma_busy;       // Any phase in progress
    wire                    bsr_dma_error;      // Parse error or bus error
    wire [AXI_ADDR_W-1:0]   bsr_dma_src_addr;   // DDR base address of BSR data

    // -------------------------------------------------------------------------
    // AXI DMA Bridge Internal Signals (2:1 Arbiter between two DMAs)
    // -------------------------------------------------------------------------
    // The bridge arbitrates between act_dma (slave 1) and bsr_dma (slave 0)
    // to share a single AXI4 master port. Round-robin with priority to
    // bsr_dma (s0) since weight loading is typically on critical path.
    //
    // Signal naming: {module}_{channel}{signal}
    //   - act_ar*: Activation DMA read address channel
    //   - act_r*:  Activation DMA read data channel
    //   - bsr_ar*: BSR DMA read address channel
    //   - bsr_r*:  BSR DMA read data channel
    //
    // act_dma → Bridge (Slave 1, lower priority)
    wire [AXI_ID_W-1:0]     act_arid;
    wire [AXI_ADDR_W-1:0]   act_araddr;
    wire [7:0]              act_arlen;
    wire [2:0]              act_arsize;
    wire [1:0]              act_arburst;
    wire                    act_arvalid;
    wire                    act_arready;
    wire [AXI_ID_W-1:0]     act_rid;
    wire [AXI_DATA_W-1:0]   act_rdata;
    wire [1:0]              act_rresp;
    wire                    act_rlast;
    wire                    act_rvalid;
    wire                    act_rready;

    // bsr_dma → Bridge (Slave 0)
    wire [AXI_ID_W-1:0]     bsr_arid;
    wire [AXI_ADDR_W-1:0]   bsr_araddr;
    wire [7:0]              bsr_arlen;
    wire [2:0]              bsr_arsize;
    wire [1:0]              bsr_arburst;
    wire                    bsr_arvalid;
    wire                    bsr_arready;
    wire [AXI_ID_W-1:0]     bsr_rid;
    wire [AXI_DATA_W-1:0]   bsr_rdata;
    wire [1:0]              bsr_rresp;
    wire                    bsr_rlast;
    wire                    bsr_rvalid;
    wire                    bsr_rready;

    // -------------------------------------------------------------------------
    // Buffer Write Interfaces (from DMAs → BRAMs)
    // -------------------------------------------------------------------------
    // These interfaces connect DMA outputs to BRAM write ports.
    // All writes are single-cycle, no back-pressure needed.
    
    // Activation Buffer Write (from act_dma → act_buffer_ram)
    // - 64-bit writes, address incremented by 8 per beat
    wire                    act_buf_we;         // Write enable
    wire [AXI_ADDR_W-1:0]   act_buf_waddr;      // Byte address
    wire [AXI_DATA_W-1:0]   act_buf_wdata;      // 8 INT8 values

    // BSR Row Pointer Write (from bsr_dma → row_ptr_bram)
    // - 32-bit entries, one per block row + 1 (CSR format)
    // - Value at row_ptr[i] is index of first non-zero block in row i
    wire                    row_ptr_we;
    wire [BRAM_ADDR_W-1:0]  row_ptr_waddr;      // Entry index (not byte addr)
    wire [31:0]             row_ptr_wdata;      // Block index

    // BSR Column Index Write (from bsr_dma → col_idx_bram)
    // - 16-bit entries, one per non-zero block
    // - Value is column index of the block in original matrix
    wire                    col_idx_we;
    wire [BRAM_ADDR_W-1:0]  col_idx_waddr;      // Entry index
    wire [15:0]             col_idx_wdata;      // Column block index

    // Weight Block Write (from bsr_dma → wgt_block_bram)
    // - 64-bit writes (8 INT8 weights per beat)
    // - Address includes block index and intra-block offset
    // - BRAM_ADDR_W+6:0 = 17 bits: [16:7] block index, [6:0] intra-block offset
    wire                    wgt_we;
    wire [BRAM_ADDR_W+6:0]  wgt_waddr;          // {block_idx, offset}
    wire [63:0]             wgt_wdata;          // 8 INT8 weights

    // -------------------------------------------------------------------------
    // Metadata Decoder Interface (BSR BRAM → Scheduler)
    // -------------------------------------------------------------------------
    // The meta_decode module provides cached access to BSR metadata.
    // It sits between bsr_scheduler and row_ptr_bram to reduce read latency
    // for frequently accessed entries (e.g., row_ptr[i] and row_ptr[i+1]).
    //
    // Request Interface (scheduler → meta_decode):
    wire                    meta_req_valid;     // Request valid
    wire [31:0]             meta_req_addr;      // Address to fetch
    wire                    meta_req_ready;     // meta_decode can accept
    
    // Response Interface (meta_decode → scheduler):
    wire                    meta_valid;         // Data valid
    wire [31:0]             meta_rdata;         // Fetched data
    wire                    meta_ready;         // Scheduler accepting data

    // Memory interface (meta_decode → row_ptr_bram)
    wire                    meta_mem_en;        // BRAM read enable
    wire [31:0]             meta_mem_addr;      // BRAM read address
    wire [31:0]             meta_mem_rdata;     // BRAM read data (1-cycle latency)

    // -------------------------------------------------------------------------
    // BSR Scheduler Interface
    // -------------------------------------------------------------------------
    // The scheduler traverses the BSR structure and generates control signals
    // for the systolic array. It operates in a weight-stationary manner:
    // 1. Load weight block → 2. Stream activations → 3. Accumulate → 4. Next block
    //
    wire                    sched_start;        // Start traversal (from CSR)
    wire                    sched_busy;         // Traversal in progress
    wire                    sched_done;         // All blocks processed
    
    // Tiled dimensions (block counts, not element counts)
    // MT = M / BLOCK_SIZE = number of output tile rows
    // KT = K / BLOCK_SIZE = number of weight tile columns
    wire [9:0]              sched_MT;           // Tile rows (max 1023)
    wire [11:0]             sched_KT;           // Tile columns (max 4095)

    // Systolic Array Control (scheduler → systolic_array)
    wire                    load_weight;        // Load weights into PE registers
    wire                    pe_en;              // Enable MAC operations
    wire                    accum_en;           // Enable accumulator update
    
    // Per-PE bypass signal for residual/skip connections
    // When bypass[i]=1, PE[i] passes input through without MAC
    // Flattened as 1D array: bypass_flat[row*N_COLS + col]
    wire [(N_ROWS*N_COLS)-1:0] bypass_flat;

    // Buffer Read Control (scheduler → buffers)
    wire                    wgt_rd_en;          // Weight buffer read enable
    wire [AXI_ADDR_W-1:0]   wgt_rd_addr;        // Weight buffer read address
    wire                    act_rd_en;          // Activation buffer read enable
    wire [AXI_ADDR_W-1:0]   act_rd_addr;        // Activation buffer read address

    // -------------------------------------------------------------------------
    // Activation Buffer Read Interface
    // -------------------------------------------------------------------------
    // Returns N_ROWS INT8 values (14 bytes = 112 bits) per read
    // - Packed as {a[13], a[12], ..., a[1], a[0]} (LSB = row 0)
    // - Fed to systolic array a_in_flat port
    wire [N_ROWS*DATA_W-1:0] act_rd_data;       // 112 bits for 14×8

    // -------------------------------------------------------------------------
    // Weight Buffer (Block Data) Read Interface
    // -------------------------------------------------------------------------
    // Returns N_COLS INT8 values (14 bytes = 112 bits) per read
    // - One row of a 14×14 weight block
    // - Fed to systolic array b_in_flat port
    wire [N_COLS*DATA_W-1:0] wgt_rd_data;       // 112 bits for 14×8

    // -------------------------------------------------------------------------
    // Sparse Systolic Array Interface
    // -------------------------------------------------------------------------
    // Output: N_ROWS × N_COLS accumulators, each ACC_W bits (32-bit)
    // - Total: 14 × 14 × 32 = 6272 bits
    // - Flattened as [row0_col0, row0_col1, ..., row13_col13]
    // - First 4 accumulators (128 bits) exposed via CSR for quick readout
    wire [N_ROWS*N_COLS*ACC_W-1:0] systolic_out_flat;  // 6272 bits

    // -------------------------------------------------------------------------
    // Performance Counters
    // -------------------------------------------------------------------------
    // These counters measure execution performance for profiling.
    // All counters are 32-bit, wrapping at ~43 seconds @ 100 MHz.
    wire [31:0] perf_total_cycles;      // Cycles from start to done
    wire [31:0] perf_active_cycles;     // Cycles with pe_en=1
    wire [31:0] perf_idle_cycles;       // Cycles waiting for data
    wire        perf_done;              // Measurement complete

    // -------------------------------------------------------------------------
    // Status Aggregation
    // -------------------------------------------------------------------------
    // Combine status from all modules for top-level outputs
    // - busy: Any module is actively processing
    // - done: Scheduler has finished all blocks
    // - error: Any DMA reported an error (address fault, bus error, etc.)
    assign busy  = act_dma_busy | bsr_dma_busy | sched_busy;
    assign done  = sched_done;
    assign error = bsr_dma_error | act_dma_error;

    // =========================================================================
    // Module Instantiations
    // =========================================================================
    // Instantiation order follows data flow:
    // 1. AXI interfaces (entry points)
    // 2. Control (CSR)
    // 3. DMA bridge (AXI arbitration)
    // 4. DMAs (data movement)
    // 5. Storage (BRAMs)
    // 6. Decode/Schedule (control flow)
    // 7. Compute (systolic array)
    // 8. Monitoring (performance)

    // -------------------------------------------------------------------------
    // 1. AXI4-Lite Slave (CSR Access from Host CPU)
    // -------------------------------------------------------------------------
    // Converts AXI4-Lite transactions to simple register read/write interface.
    // Handles AXI protocol (ready/valid handshakes, write response).
    //
    // Timing:
    //   - Write: 2 cycles (awvalid + wvalid → bvalid)
    //   - Read: 2 cycles (arvalid → rvalid)
    //
    // Note: Address is truncated to CSR_ADDR_W bits. Host driver must ensure
    // accesses are within the configured address range.
    axi_lite_slave #(
        .CSR_ADDR_WIDTH(CSR_ADDR_W),
        .CSR_DATA_WIDTH(32)
    ) u_axi_lite_slave (
        .clk            (clk),
        .rst_n          (rst_n),
        // AXI4-Lite Write
        .s_axi_awaddr   (s_axi_awaddr[CSR_ADDR_W-1:0]),
        .s_axi_awprot   (s_axi_awprot),
        .s_axi_awvalid  (s_axi_awvalid),
        .s_axi_awready  (s_axi_awready),
        .s_axi_wdata    (s_axi_wdata),
        .s_axi_wstrb    (s_axi_wstrb),
        .s_axi_wvalid   (s_axi_wvalid),
        .s_axi_wready   (s_axi_wready),
        .s_axi_bresp    (s_axi_bresp),
        .s_axi_bvalid   (s_axi_bvalid),
        .s_axi_bready   (s_axi_bready),
        // AXI4-Lite Read
        .s_axi_araddr   (s_axi_araddr[CSR_ADDR_W-1:0]),
        .s_axi_arprot   (s_axi_arprot),
        .s_axi_arvalid  (s_axi_arvalid),
        .s_axi_arready  (s_axi_arready),
        .s_axi_rdata    (s_axi_rdata),
        .s_axi_rresp    (s_axi_rresp),
        .s_axi_rvalid   (s_axi_rvalid),
        .s_axi_rready   (s_axi_rready),
        // CSR Interface
        .csr_addr       (csr_addr),
        .csr_wen        (csr_wen),
        .csr_ren        (csr_ren),
        .csr_wdata      (csr_wdata),
        .csr_rdata      (csr_rdata),
        .axi_error      ()
    );

    // -------------------------------------------------------------------------
    // 2. CSR Module (Configuration & Control Registers)
    // -------------------------------------------------------------------------
    // Central register bank for accelerator configuration and status.
    // See csr.sv for full address map documentation.
    //
    // Key Registers:
    //   - CTRL (0x00): Start/abort control (W1P bits)
    //   - STATUS (0x04): Busy/done/error flags (RO)
    //   - M/N/K (0x10-0x18): Matrix dimensions
    //   - DMA_SRC_ADDR (0x40): BSR data base address
    //   - ACT_SRC_ADDR (0x48): Activation data base address
    //   - PERF_* (0x80+): Performance counters
    //
    // Unused outputs are left unconnected for future features.
    csr #(
        .ADDR_W(CSR_ADDR_W)
    ) u_csr (
        .clk                    (clk),
        .rst_n                  (rst_n),
        // CSR Bus
        .csr_wen                (csr_wen),
        .csr_ren                (csr_ren),
        .csr_addr               (csr_addr),
        .csr_wdata              (csr_wdata),
        .csr_rdata              (csr_rdata),
        // Status Inputs
        .core_busy              (busy),
        .core_done_tile_pulse   (sched_done),
        .core_bank_sel_rd_A     (1'b0),
        .core_bank_sel_rd_B     (1'b0),
        .rx_illegal_cmd         (1'b0),  // No UART, no illegal commands
        // Control Outputs
        .start_pulse            (start_pulse),
        .abort_pulse            (abort_pulse),
        .irq_en                 (),
        // Matrix Dimensions
        .M                      (cfg_M),
        .N                      (cfg_N),
        .K                      (cfg_K),
        // Tile Sizes (not used in sparse mode, but kept for compatibility)
        .Tm                     (),
        .Tn                     (),
        .Tk                     (),
        .m_idx                  (),
        .n_idx                  (),
        .k_idx                  (),
        // Bank Selection (legacy, unused in sparse)
        .bank_sel_wr_A          (),
        .bank_sel_wr_B          (),
        .bank_sel_rd_A          (),
        .bank_sel_rd_B          (),
        // Scaling Factors (for future quantization)
        .Sa_bits                (),
        .Sw_bits                (),
        // Performance Counters
        .perf_total_cycles      (perf_total_cycles),
        .perf_active_cycles     (perf_active_cycles),
        .perf_idle_cycles       (perf_idle_cycles),
        .perf_cache_hits        (32'd0),  // TODO: Connect from meta_decode
        .perf_cache_misses      (32'd0),
        .perf_decode_count      (32'd0),
        // Result Data (first 4 accumulators for quick read)
        .result_data            (systolic_out_flat[127:0]),
        // DMA Control/Status
        .dma_busy_in            (act_dma_busy | bsr_dma_busy),
        .dma_done_in            (act_dma_done & bsr_dma_done),
        .dma_bytes_xferred_in   (32'd0),  // TODO: Add counters
        .dma_src_addr           (cfg_bsr_src_addr),
        .dma_dst_addr           (),       // Not used (read-only DMAs)
        .dma_xfer_len           (),
        .dma_start_pulse        (bsr_dma_start),
        // Activation DMA
        .act_dma_src_addr       (cfg_act_src_addr),
        .act_dma_len            (cfg_act_xfer_len),
        .act_dma_start_pulse    (cfg_act_dma_start)
    );

    // -------------------------------------------------------------------------
    // CSR → DMA/Scheduler Signal Routing
    // -------------------------------------------------------------------------
    // Route CSR outputs to the appropriate modules. These are direct
    // combinational connections (no pipelining).
    
    // BSR DMA source address (base of BSR structure in DDR)
    assign bsr_dma_src_addr = cfg_bsr_src_addr;
    // Activation DMA control signals
    assign act_dma_start    = cfg_act_dma_start;
    assign act_dma_src_addr = cfg_act_src_addr;
    assign act_dma_xfer_len = cfg_act_xfer_len;

    // Scheduler control signals
    // - sched_start: Single-cycle pulse to begin BSR traversal
    // - sched_MT: Block rows = M / BLOCK_SIZE (right-shift by 3 since BLOCK_SIZE ≠ 8)
    //   NOTE: The shift by 3 assumes BLOCK_SIZE=8. For 14×14 blocks, this should
    //   be division by 14, but hardware division is expensive. Either:
    //   1. Use fixed block sizes that are powers of 2
    //   2. Pre-compute MT/KT on the host and store in CSR
    //   Current implementation uses shift-by-3 as placeholder.
    assign sched_start = start_pulse;
    assign sched_MT    = cfg_M[9:0] >> 3;   // TODO: Fix for non-power-of-2 block size
    assign sched_KT    = cfg_K[11:0] >> 3;  // TODO: Fix for non-power-of-2 block size

    // -------------------------------------------------------------------------
    // 3. AXI DMA Bridge (Arbitrates act_dma and bsr_dma to single AXI Master)
    // -------------------------------------------------------------------------
    // Two-slave AXI4 read arbiter. Multiplexes read address and data channels
    // from act_dma (s1) and bsr_dma (s0) to a single master port.
    //
    // Arbitration Policy:
    //   - Round-robin when both requesting
    //   - s0 (bsr_dma) has slight priority (checked first)
    //
    // ID Routing:
    //   - s0 transactions use ID bit [0] = 0
    //   - s1 transactions use ID bit [0] = 1
    //   - Allows routing responses back to correct slave
    //
    // Bandwidth Considerations:
    //   - HP port: ~2 GB/s effective bandwidth
    //   - Two DMAs rarely contend (different phases of operation)
    //   - bsr_dma runs first (load weights), then act_dma (stream activations)
    axi_dma_bridge #(
        .DATA_WIDTH (AXI_DATA_W),
        .ADDR_WIDTH (AXI_ADDR_W),
        .ID_WIDTH   (AXI_ID_W)
    ) u_axi_dma_bridge (
        .clk            (clk),
        .rst_n          (rst_n),
        // Slave 0: BSR DMA
        .s0_arid        (bsr_arid),
        .s0_araddr      (bsr_araddr),
        .s0_arlen       (bsr_arlen),
        .s0_arsize      (bsr_arsize),
        .s0_arburst     (bsr_arburst),
        .s0_arvalid     (bsr_arvalid),
        .s0_arready     (bsr_arready),
        .s0_rid         (bsr_rid),
        .s0_rdata       (bsr_rdata),
        .s0_rresp       (bsr_rresp),
        .s0_rlast       (bsr_rlast),
        .s0_rvalid      (bsr_rvalid),
        .s0_rready      (bsr_rready),
        // Slave 1: Activation DMA
        .s1_arid        (act_arid),
        .s1_araddr      (act_araddr),
        .s1_arlen       (act_arlen),
        .s1_arsize      (act_arsize),
        .s1_arburst     (act_arburst),
        .s1_arvalid     (act_arvalid),
        .s1_arready     (act_arready),
        .s1_rid         (act_rid),
        .s1_rdata       (act_rdata),
        .s1_rresp       (act_rresp),
        .s1_rlast       (act_rlast),
        .s1_rvalid      (act_rvalid),
        .s1_rready      (act_rready),
        // Master to DDR
        .m_arid         (m_axi_arid),
        .m_araddr       (m_axi_araddr),
        .m_arlen        (m_axi_arlen),
        .m_arsize       (m_axi_arsize),
        .m_arburst      (m_axi_arburst),
        .m_arvalid      (m_axi_arvalid),
        .m_arready      (m_axi_arready),
        .m_rid          (m_axi_rid),
        .m_rdata        (m_axi_rdata),
        .m_rresp        (m_axi_rresp),
        .m_rlast        (m_axi_rlast),
        .m_rvalid       (m_axi_rvalid),
        .m_rready       (m_axi_rready)
    );

    // -------------------------------------------------------------------------
    // 4. Activation DMA (Loads Dense Activations from DDR → act_buffer)
    // -------------------------------------------------------------------------
    // Performs burst reads from DDR to load activation vectors.
    // Activations are stored as contiguous INT8 arrays in row-major order.
    //
    // Transfer Flow:
    //   1. Host writes cfg_act_src_addr (DDR address) and cfg_act_xfer_len
    //   2. Host pulses cfg_act_dma_start
    //   3. DMA issues 16-beat bursts until xfer_len bytes transferred
    //   4. DMA asserts act_dma_done when complete
    //
    // STREAM_ID = 1: Distinguishes from BSR DMA in AXI ID field for debugging
    // BURST_LEN = 15: 16 beats × 8 bytes = 128 bytes per burst
    act_dma #(
        .AXI_ADDR_W (AXI_ADDR_W),
        .AXI_DATA_W (AXI_DATA_W),
        .AXI_ID_W   (AXI_ID_W),
        .STREAM_ID  (1),            // ID = 1 for Act DMA (s1 of bridge)
        .BURST_LEN  (8'd15)         // 16-beat bursts (128 bytes)
    ) u_act_dma (
        .clk                (clk),
        .rst_n              (rst_n),
        // Control
        .start              (act_dma_start),
        .src_addr           (act_dma_src_addr),
        .transfer_length    (act_dma_xfer_len),
        .done               (act_dma_done),
        .busy               (act_dma_busy),
        .error              (act_dma_error),
        // AXI Master (to Bridge)
        .m_axi_arid         (act_arid),
        .m_axi_araddr       (act_araddr),
        .m_axi_arlen        (act_arlen),
        .m_axi_arsize       (act_arsize),
        .m_axi_arburst      (act_arburst),
        .m_axi_arvalid      (act_arvalid),
        .m_axi_arready      (act_arready),
        .m_axi_rid          (act_rid),
        .m_axi_rdata        (act_rdata),
        .m_axi_rresp        (act_rresp),
        .m_axi_rlast        (act_rlast),
        .m_axi_rvalid       (act_rvalid),
        .m_axi_rready       (act_rready),
        // Buffer Write Interface
        .act_we             (act_buf_we),
        .act_addr           (act_buf_waddr),
        .act_wdata          (act_buf_wdata)
    );

    // -------------------------------------------------------------------------
    // 5. BSR DMA (Loads Sparse Weights from DDR → Metadata BRAMs)
    // -------------------------------------------------------------------------
    // Parses BSR (Block Sparse Row) format and populates three BRAMs:
    //   1. row_ptr_bram: CSR-style row pointers (32-bit each)
    //   2. col_idx_bram: Column indices of non-zero blocks (16-bit each)
    //   3. wgt_block_bram: Weight data for non-zero blocks (14×14 INT8 each)
    //
    // BSR Format in DDR (see docs/architecture/SPARSITY_FORMAT.md):
    //   Offset 0x00: num_rows (32-bit)
    //   Offset 0x04: num_cols (32-bit)
    //   Offset 0x08: nnz_blocks (32-bit)
    //   Offset 0x0C: block_size (32-bit, always 14 for this design)
    //   Offset 0x10: row_ptr[0..num_rows] (num_rows+1 × 32-bit)
    //   After row_ptr: col_idx[0..nnz_blocks-1] (nnz_blocks × 16-bit)
    //   After col_idx (aligned): blocks[0..nnz_blocks-1] (nnz_blocks × 196 bytes)
    //
    // STREAM_ID = 0: Primary DMA, checked first in bridge arbiter
    bsr_dma #(
        .AXI_ADDR_W  (AXI_ADDR_W),
        .AXI_DATA_W  (AXI_DATA_W),
        .AXI_ID_W    (AXI_ID_W),
        .STREAM_ID   (0),           // ID = 0 for BSR DMA (s0 of bridge)
        .BRAM_ADDR_W (BRAM_ADDR_W),
        .BURST_LEN   (8'd15)        // 16-beat bursts
    ) u_bsr_dma (
        .clk                (clk),
        .rst_n              (rst_n),
        // Control
        .start              (bsr_dma_start),
        .src_addr           (bsr_dma_src_addr),
        .done               (bsr_dma_done),
        .busy               (bsr_dma_busy),
        .error              (bsr_dma_error),
        // AXI Master (to Bridge)
        .m_axi_arid         (bsr_arid),
        .m_axi_araddr       (bsr_araddr),
        .m_axi_arlen        (bsr_arlen),
        .m_axi_arsize       (bsr_arsize),
        .m_axi_arburst      (bsr_arburst),
        .m_axi_arvalid      (bsr_arvalid),
        .m_axi_arready      (bsr_arready),
        .m_axi_rid          (bsr_rid),
        .m_axi_rdata        (bsr_rdata),
        .m_axi_rresp        (bsr_rresp),
        .m_axi_rlast        (bsr_rlast),
        .m_axi_rvalid       (bsr_rvalid),
        .m_axi_rready       (bsr_rready),
        // BRAM Write Interfaces
        .row_ptr_we         (row_ptr_we),
        .row_ptr_addr       (row_ptr_waddr),
        .row_ptr_wdata      (row_ptr_wdata),
        .col_idx_we         (col_idx_we),
        .col_idx_addr       (col_idx_waddr),
        .col_idx_wdata      (col_idx_wdata),
        .wgt_we             (wgt_we),
        .wgt_addr           (wgt_waddr),
        .wgt_wdata          (wgt_wdata)
    );

    // -------------------------------------------------------------------------
    // 6. BSR Metadata BRAMs (Inferred Dual-Port RAMs)
    // -------------------------------------------------------------------------
    // These are simple dual-port RAMs with one write port (from DMA) and
    // one read port (from scheduler/meta_decode). Xilinx Vivado will infer
    // RAMB36E1 (36Kb) or RAMB18E1 (18Kb) primitives.
    //
    // Memory Sizing:
    //   - row_ptr_bram: 1024 × 32-bit = 32Kb → 1× RAMB36 or 2× RAMB18
    //   - col_idx_bram: 1024 × 16-bit = 16Kb → 1× RAMB18
    //   - wgt_block_bram: 1024 × 64-bit = 64Kb → 2× RAMB36
    //   Total: ~112Kb (~3-4 BRAM36 equivalent)
    //
    // Timing:
    //   - Write: Captured on rising edge when *_we=1
    //   - Read: 1-cycle latency (address registered, data available next cycle)
    //
    // Note: These are instantiated as behavioral RTL. For timing-critical
    // designs, consider using Xilinx XPM_MEMORY macros.
    
    // -------------------------------------------------------------------------
    // 6a. Row Pointer BRAM (32-bit entries, CSR format)
    // -------------------------------------------------------------------------
    // row_ptr[i] = index of first non-zero block in row i
    // row_ptr[num_rows] = total number of non-zero blocks
    reg [31:0] row_ptr_bram [0:(1<<BRAM_ADDR_W)-1];
    reg [31:0] row_ptr_rdata_r;

    always @(posedge clk) begin
        if (row_ptr_we)
            row_ptr_bram[row_ptr_waddr] <= row_ptr_wdata;
        // Read port: address from meta_decode module
        row_ptr_rdata_r <= row_ptr_bram[meta_mem_addr[BRAM_ADDR_W-1:0]];
    end

    // -------------------------------------------------------------------------
    // 6b. Column Index BRAM (16-bit entries)
    // -------------------------------------------------------------------------
    // col_idx[j] = column block index for j-th non-zero block
    // Used by scheduler to compute activation address offset
    reg [15:0] col_idx_bram [0:(1<<BRAM_ADDR_W)-1];
    reg [15:0] col_idx_rdata_r;

    always @(posedge clk) begin
        if (col_idx_we)
            col_idx_bram[col_idx_waddr] <= col_idx_wdata;
        // Read port: Currently unused (scheduler reads via meta_decode)
        // TODO: Add read interface if direct col_idx access needed
    end

    // -------------------------------------------------------------------------
    // 6c. Weight Block BRAM (64-bit entries)
    // -------------------------------------------------------------------------
    // Stores 14×14 = 196 INT8 weights per block, packed 8 per 64-bit word.
    // Each block occupies 196/8 = 24.5 → 25 words (last word half-used).
    //
    // Address format: wgt_waddr[BRAM_ADDR_W+6:0] = {block_idx, word_offset}
    // - block_idx: Which non-zero block (0 to nnz_blocks-1)
    // - word_offset: 0-24 within block (5 bits, but only 25 used)
    //
    // Read interface uses wgt_rd_addr from scheduler.
    reg [63:0] wgt_block_bram [0:(1<<BRAM_ADDR_W)-1];
    reg [63:0] wgt_block_rdata_r;

    always @(posedge clk) begin
        if (wgt_we)
            wgt_block_bram[wgt_waddr[BRAM_ADDR_W-1:0]] <= wgt_wdata;
        // Read port: Scheduler requests weight data
        wgt_block_rdata_r <= wgt_block_bram[wgt_rd_addr[BRAM_ADDR_W-1:0]];
    end

    // Connect row_ptr read data to metadata interface
    // Future: Mux between row_ptr and col_idx based on meta_mem_addr MSB
    assign meta_mem_rdata = row_ptr_rdata_r;

    // -------------------------------------------------------------------------
    // 7. Metadata Decoder (Cache for BSR Metadata)
    // -------------------------------------------------------------------------
    // Provides cached access to BSR row pointers for the scheduler.
    // The scheduler frequently reads row_ptr[i] and row_ptr[i+1] to determine
    // the range of non-zero blocks in each row. Caching reduces BRAM access
    // latency and improves throughput.
    //
    // Cache Architecture:
    //   - Direct-mapped, 64 entries
    //   - 1-cycle hit, 2-cycle miss (BRAM latency)
    //   - Tag comparison on meta_req_addr
    //
    // Future Enhancement: Add col_idx caching for faster column lookups.
    meta_decode #(
        .DATA_WIDTH  (32),          // 32-bit row pointers
        .CACHE_DEPTH (64)           // 64-entry cache (covers 64 rows)
    ) u_meta_decode (
        .clk            (clk),
        .rst_n          (rst_n),
        // Scheduler Interface
        .req_valid      (meta_req_valid),
        .req_addr       (meta_req_addr),
        .req_ready      (meta_req_ready),
        // Memory Interface
        .mem_en         (meta_mem_en),
        .mem_addr       (meta_mem_addr),
        .mem_rdata      (meta_mem_rdata),
        // Output to Scheduler
        .meta_valid     (meta_valid),
        .meta_rdata     (meta_rdata),
        .meta_ready     (meta_ready)
    );

    // -------------------------------------------------------------------------
    // 8. BSR Scheduler (Traverses Sparse Blocks)
    // -------------------------------------------------------------------------
    // Iterates through the BSR structure and generates control signals for
    // the systolic array. Implements weight-stationary dataflow:
    //
    // For each block row (m_tile = 0 to MT-1):
    //   ptr_start = row_ptr[m_tile]
    //   ptr_end   = row_ptr[m_tile + 1]
    //   For each non-zero block (blk = ptr_start to ptr_end-1):
    //     col = col_idx[blk]
    //     1. Load weight block[blk] into systolic array (load_weight=1)
    //     2. Stream 14 activation rows (pe_en=1)
    //     3. Accumulate results (accum_en=1)
    //   End block loop (results stay in accumulators)
    // End row loop → done
    //
    // BLOCK_SIZE must match systolic array dimensions (N_ROWS = N_COLS = 14).
    bsr_scheduler #(
        .M_W        (10),           // Tile row index width (up to 1023 rows)
        .N_W        (10),           // Tile col index width (up to 1023 cols)
        .K_W        (12),           // Tile K index width (up to 4095 cols)
        .ADDR_W     (AXI_ADDR_W),   // Address width for buffer access
        .BLOCK_SIZE (N_ROWS)        // Block size = systolic array dimension
    ) u_bsr_scheduler (
        .clk            (clk),
        .rst_n          (rst_n),
        // Control
        .start          (sched_start),
        .abort          (abort_pulse),
        .busy           (sched_busy),
        .done           (sched_done),
        // Configuration
        .MT             (sched_MT),
        .KT             (sched_KT),
        // Metadata Interface
        .meta_raddr     (meta_req_addr),
        .meta_ren       (meta_req_valid),
        .meta_rdata     (meta_rdata),
        .meta_rvalid    (meta_valid),
        .meta_ready     (meta_ready),
        // Buffer Interfaces
        .wgt_rd_en      (wgt_rd_en),
        .wgt_addr       (wgt_rd_addr),
        .act_rd_en      (act_rd_en),
        .act_addr       (act_rd_addr),
        // Systolic Control
        .load_weight    (load_weight),
        .pe_en          (pe_en),
        .accum_en       (accum_en),
        .bypass_out     (bypass_flat)  // Per-PE bypass for residual connections
    );

    // -------------------------------------------------------------------------
    // 9. Activation Buffer (Dense Activation Storage)
    // -------------------------------------------------------------------------
    // Simple dual-port RAM for activation data. DMA writes, scheduler reads.
    //
    // Note: This is a simplified implementation using behavioral RTL.
    // For the full design, use act_buffer.sv module with ping-pong banks
    // and clock gating. This inline version is for integration testing.
    //
    // Capacity: 1024 × 64 bits = 8 KB (holds ~8192 INT8 activations)
    // For a 14×14 tile, need 196 bytes. This buffer can hold ~40 tiles.
    reg [AXI_DATA_W-1:0] act_buffer_ram [0:(1<<BRAM_ADDR_W)-1];
    reg [AXI_DATA_W-1:0] act_rd_data_r;

    always @(posedge clk) begin
        // Write port: DMA stores incoming activations
        if (act_buf_we)
            act_buffer_ram[act_buf_waddr[BRAM_ADDR_W-1:0]] <= act_buf_wdata;
        // Read port: Scheduler fetches activations for systolic array
        if (act_rd_en)
            act_rd_data_r <= act_buffer_ram[act_rd_addr[BRAM_ADDR_W-1:0]];
    end

    // Extract N_ROWS (14) INT8 values from 64-bit read data
    // Only lower 112 bits (14 × 8) are used; upper 16 bits ignored
    assign act_rd_data = act_rd_data_r[N_ROWS*DATA_W-1:0];

    // -------------------------------------------------------------------------
    // 10. Weight Read Path (Block data from wgt_block_bram)
    // -------------------------------------------------------------------------
    // Extracts one row of weights (14 INT8 values) from 64-bit BRAM output.
    // The scheduler controls which row of the current block is read.
    //
    // Note: Full 14×14 block needs 196 bytes. With 64-bit reads, need 25 reads
    // to fetch entire block (3 bytes wasted in last read).
    assign wgt_rd_data = wgt_block_rdata_r[N_COLS*DATA_W-1:0];

    // -------------------------------------------------------------------------
    // 11. Sparse Systolic Array (2D PE Array with Skip-Zero Logic)
    // -------------------------------------------------------------------------
    // Core compute unit: 14×14 grid of Processing Elements (PEs).
    // Each PE performs: acc[i][j] += a[i] × w[j]
    //
    // Weight-Stationary Dataflow:
    //   1. Weights loaded once per block (load_weight=1)
    //   2. Activations streamed through (pe_en=1 for 14 cycles)
    //   3. Results accumulate across K dimension
    //
    // Sparse Optimization:
    //   - Zero blocks are skipped entirely (no MAC cycles)
    //   - Per-PE bypass (bypass_flat) for future residual connections
    //
    // Resource Usage: 196 DSP48E1 slices (one per PE)
    systolic_array_sparse #(
        .N_ROWS (N_ROWS),           // 14 rows
        .N_COLS (N_COLS),           // 14 columns
        .DATA_W (DATA_W),           // 8-bit inputs
        .ACC_W  (ACC_W)             // 32-bit accumulators
    ) u_systolic_sparse (
        .clk            (clk),
        .rst_n          (rst_n),
        // Control
        .block_valid    (pe_en),            // Enable MAC operations
        .load_weight    (load_weight),      // Load weights into PE registers
        .bypass_flat    (bypass_flat),      // Per-PE bypass (for residuals)
        // Data
        .a_in_flat      (act_rd_data),      // 14 × 8-bit activations
        .b_in_flat      (wgt_rd_data),      // 14 × 8-bit weights
        // Output
        .c_out_flat     (systolic_out_flat) // 196 × 32-bit accumulators
    );

    // -------------------------------------------------------------------------
    // 12. Performance Monitor
    // -------------------------------------------------------------------------
    // Tracks execution cycles for profiling and optimization.
    //
    // Counters:
    //   - total_cycles: Cycles from start_pulse to sched_done
    //   - active_cycles: Cycles with busy=1 (actual computation)
    //   - idle_cycles: Cycles waiting for memory/stalls
    //
    // These counters are readable via CSR PERF_* registers.
    // Host can compute:
    //   - Utilization = active_cycles / total_cycles
    //   - Throughput = (M × N × K × 2) / total_cycles ops/cycle
    //
    // TODO: Add cache hit/miss counters from meta_decode for memory analysis
    perf #(
        .COUNTER_WIDTH(32)          // 32-bit counters (43 sec @ 100 MHz)
    ) u_perf (
        .clk                    (clk),
        .rst_n                  (rst_n),
        // Control
        .start_pulse            (start_pulse),  // Begin measurement
        .done_pulse             (sched_done),   // End measurement
        .busy_signal            (busy),         // Active computation
        // Cache Stats (TODO: Connect from meta_decode)
        .meta_cache_hits        (32'd0),
        .meta_cache_misses      (32'd0),
        .meta_decode_cycles     (32'd0),
        // Outputs (to CSR)
        .total_cycles_count     (perf_total_cycles),
        .active_cycles_count    (perf_active_cycles),
        .idle_cycles_count      (perf_idle_cycles),
        .cache_hit_count        (),             // Future use
        .cache_miss_count       (),             // Future use
        .decode_count           (),             // Future use
        .measurement_done       (perf_done)
    );

endmodule

`default_nettype wire
