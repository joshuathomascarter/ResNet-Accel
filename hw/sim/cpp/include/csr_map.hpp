/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                            CSR_MAP.HPP                                    ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  Control and Status Register (CSR) address map definitions for the       ║
 * ║  INT8 Sparse Systolic Array Accelerator on Zynq-7020.                    ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  CRITICAL: These definitions MUST MATCH hw/rtl/control/csr.sv!           ║
 * ║  Any mismatch will cause silent data corruption or control failures.     ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  Memory-Mapped Address Ranges (from ACCEL_BASE_ADDR):                    ║
 * ║    0x00 - 0x3F: Control & Configuration Registers                        ║
 * ║                 - Start/abort control, matrix dimensions, tile config    ║
 * ║    0x40 - 0x5F: Performance Counters (Read-Only)                          ║
 * ║                 - Cycle counts, cache stats, utilization metrics         ║
 * ║    0x80 - 0x8F: Result Registers (Read-Only)                              ║
 * ║                 - First 4 accumulator outputs for quick readout          ║
 * ║    0x90 - 0xBF: DMA Configuration                                        ║
 * ║                 - Source/dest addresses, transfer lengths, control       ║
 * ║                                                                           ║
 * ║  Register Types:                                                          ║
 * ║    R/W:   Standard read/write register                                   ║
 * ║    RO:    Read-only (writes ignored)                                     ║
 * ║    W1P:   Write-1-Pulse (write 1 to trigger single-cycle pulse)          ║
 * ║    W1C:   Write-1-Clear (write 1 to clear status bit)                    ║
 * ║                                                                           ║
 * ║  Usage from Linux Userspace:                                              ║
 * ║    int fd = open("/dev/mem", O_RDWR | O_SYNC);                           ║
 * ║    void* base = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED,       ║
 * ║                      fd, ACCEL_BASE_ADDR);                                ║
 * ║    volatile uint32_t* csr = (volatile uint32_t*)base;                    ║
 * ║    csr[DIMS_M/4] = matrix_m;  // Write dimension                         ║
 * ║    csr[CTRL/4] = CTRL_START;  // Trigger computation                     ║
 * ║    while (csr[STATUS/4] & STATUS_BUSY) { }  // Poll busy                 ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#ifndef CSR_MAP_HPP
#define CSR_MAP_HPP

#include <cstdint>

namespace resnet_accel {
namespace csr {

// =============================================================================
// CONTROL & CONFIGURATION REGISTERS (0x00 - 0x3F)
// =============================================================================
// These registers configure the accelerator before computation begins.
// All are R/W unless noted.

/// CTRL (0x00): Control Register
/// Bits:
///   [0] start:   W1P - Write 1 to begin computation. Self-clears.
///   [1] abort:   W1P - Write 1 to abort current operation. Self-clears.
///   [2] irq_en:  R/W - Enable interrupt on completion (TODO: implement IRQ)
constexpr uint32_t CTRL         = 0x00;

/// DIMS_M (0x04): Matrix M Dimension (output rows)
/// Range: 1 to 65535 (16-bit effective, upper bits ignored)
/// For tiled operation: Total output rows, scheduler computes tiles.
constexpr uint32_t DIMS_M       = 0x04;

/// DIMS_N (0x08): Matrix N Dimension (output columns)
/// Range: 1 to 65535
/// For ResNet-18: Typically 1 (single output per sample)
constexpr uint32_t DIMS_N       = 0x08;

/// DIMS_K (0x0C): Matrix K Dimension (inner/reduction dimension)
/// Range: 1 to 65535
/// For conv2d: in_channels × kernel_h × kernel_w
constexpr uint32_t DIMS_K       = 0x0C;

/// TILES_Tm (0x10): Tile Size M (rows per tile)
/// Default: 14 (matches systolic array rows for weight-stationary)
constexpr uint32_t TILES_Tm     = 0x10;

/// TILES_Tn (0x14): Tile Size N (columns per tile)
/// Default: 14 (matches systolic array columns)
constexpr uint32_t TILES_Tn     = 0x14;

/// TILES_Tk (0x18): Tile Size K (inner dimension per tile)
/// Default: 14 for BSR block size
constexpr uint32_t TILES_Tk     = 0x18;

/// INDEX_m (0x1C): Current Tile Index M
/// For multi-tile operations, tracks progress. Usually read-only in practice.
constexpr uint32_t INDEX_m      = 0x1C;

/// INDEX_n (0x20): Current Tile Index N
constexpr uint32_t INDEX_n      = 0x20;

/// INDEX_k (0x24): Current Tile Index K
constexpr uint32_t INDEX_k      = 0x24;

/// BUFF (0x28): Buffer Bank Select
/// Bits:
///   [0] wrA:  R/W - Write to bank A enable
///   [1] wrB:  R/W - Write to bank B enable
///   [8] rdA:  RO  - Currently reading from bank A
///   [9] rdB:  RO  - Currently reading from bank B
/// For ping-pong: Write to one bank while reading from other.
constexpr uint32_t BUFF         = 0x28;

/// SCALE_Sa (0x2C): Activation Scale Factor
/// IEEE 754 float32 bits. For INT8 quantization: result = int32 * Sa * Sw
constexpr uint32_t SCALE_Sa     = 0x2C;

/// SCALE_Sw (0x30): Weight Scale Factor
/// IEEE 754 float32 bits.
constexpr uint32_t SCALE_Sw     = 0x30;

/// STATUS (0x3C): Status Register
/// Bits:
///   [0] busy:         RO   - Core is processing (DMA or compute)
///   [1] done_tile:    W1C  - Tile computation complete. Write 1 to clear.
///   [9] err_illegal:  W1C  - Illegal command received. Write 1 to clear.
constexpr uint32_t STATUS       = 0x3C;

// =============================================================================
// PERFORMANCE COUNTERS (0x40 - 0x5F) — Read Only
// =============================================================================
// These counters measure execution performance. Latched on computation complete.
// 32-bit counters wrap at ~43 seconds @ 100 MHz.

/// PERF_TOTAL (0x40): Total cycles from start to done
/// Use for end-to-end latency measurement.
constexpr uint32_t PERF_TOTAL       = 0x40;

/// PERF_ACTIVE (0x44): Cycles with busy=1
/// Utilization = PERF_ACTIVE / PERF_TOTAL
constexpr uint32_t PERF_ACTIVE      = 0x44;

/// PERF_IDLE (0x48): Cycles with busy=0 (stalls, memory waits)
/// Optimization target: Minimize this for higher throughput.
constexpr uint32_t PERF_IDLE        = 0x48;

/// PERF_CACHE_HITS (0x4C): Metadata cache hits
/// High value indicates good cache utilization.
constexpr uint32_t PERF_CACHE_HITS  = 0x4C;

/// PERF_CACHE_MISSES (0x50): Metadata cache misses
/// Each miss adds ~2 cycles latency. Optimize by sequential row access.
constexpr uint32_t PERF_CACHE_MISSES = 0x50;

/// PERF_DECODE_COUNT (0x54): Metadata decode operations
/// Should equal nnz_blocks for correct operation.
constexpr uint32_t PERF_DECODE_COUNT = 0x54;

// =============================================================================
// RESULT REGISTERS (0x80 - 0x8F) — Read Only
// =============================================================================
// First 4 accumulators exposed for quick result readout.
// For full output, use DMA to read output buffer.

/// RESULT_0-3: First 4 accumulator outputs (INT32)
/// Contains c_out[0..3] from systolic array.
constexpr uint32_t RESULT_0     = 0x80;  // c_out[0]
constexpr uint32_t RESULT_1     = 0x84;  // c_out[1]
constexpr uint32_t RESULT_2     = 0x88;  // c_out[2]
constexpr uint32_t RESULT_3     = 0x8C;  // c_out[3]

// =============================================================================
// BSR DMA CONTROL (0x90 - 0xBF)
// =============================================================================
// Controls the BSR (Block Sparse Row) weight DMA engine.
// Loads sparse weights from DDR into internal BRAMs.

/// DMA_SRC_ADDR (0x90): DDR source address for BSR data
/// Must be 8-byte aligned for efficient AXI bursts.
constexpr uint32_t DMA_SRC_ADDR     = 0x90;

/// DMA_DST_ADDR (0x94): Destination address (buffer select)
/// Bit encoding TBD - typically selects internal buffer bank.
constexpr uint32_t DMA_DST_ADDR     = 0x94;

/// DMA_XFER_LEN (0x98): Transfer length in bytes
/// For BSR: header(16) + row_ptr((M/14+1)*4) + col_idx(nnz*2) + blocks(nnz*196)
constexpr uint32_t DMA_XFER_LEN     = 0x98;

/// DMA_CTRL (0x9C): DMA Control Register
/// Bits:
///   [0] start: W1P - Trigger DMA transfer
///   [1] busy:  RO  - Transfer in progress
///   [2] done:  W1C - Transfer complete
constexpr uint32_t DMA_CTRL         = 0x9C;

/// DMA_BYTES_XFERRED (0xB8): Bytes transferred counter (RO)
/// For debugging/progress monitoring.
constexpr uint32_t DMA_BYTES_XFERRED = 0xB8;

// =============================================================================
// ACTIVATION DMA CONTROL (0xA0 - 0xAF)
// =============================================================================
// Controls the activation DMA engine.
// Loads dense activation vectors from DDR.

/// ACT_DMA_SRC_ADDR (0xA0): DDR source address for activations
/// Must be 8-byte aligned.
constexpr uint32_t ACT_DMA_SRC_ADDR = 0xA0;

/// ACT_DMA_LEN (0xA4): Activation transfer length in bytes
/// For a tile: Tm × Tk bytes (e.g., 14 × 14 = 196 bytes)
constexpr uint32_t ACT_DMA_LEN      = 0xA4;

/// ACT_DMA_CTRL (0xA8): Activation DMA Control Register
/// Same bit encoding as DMA_CTRL.
constexpr uint32_t ACT_DMA_CTRL     = 0xA8;

// =============================================================================
// CONTROL REGISTER BIT DEFINITIONS (CTRL @ 0x00)
// =============================================================================

/// CTRL_START: Write 1 to begin computation
/// Clears automatically after pulse generated.
constexpr uint32_t CTRL_START       = (1 << 0);

/// CTRL_ABORT: Write 1 to abort current operation
/// Resets internal state machines. Use for error recovery.
constexpr uint32_t CTRL_ABORT       = (1 << 1);

/// CTRL_IRQ_EN: Enable interrupt on completion
/// When set, IRQ output asserted on done. Requires IRQ handler setup.
constexpr uint32_t CTRL_IRQ_EN      = (1 << 2);

// =============================================================================
// STATUS REGISTER BIT DEFINITIONS (STATUS @ 0x3C)
// =============================================================================

/// STATUS_BUSY: Accelerator is processing (DMA or compute)
/// Poll this to wait for completion: while (STATUS & STATUS_BUSY) {}
constexpr uint32_t STATUS_BUSY          = (1 << 0);

/// STATUS_DONE_TILE: Tile computation complete
/// Write 1 to clear. Check after busy goes low.
constexpr uint32_t STATUS_DONE_TILE     = (1 << 1);

/// STATUS_ERR_ILLEGAL: Illegal command error
/// Write 1 to clear. Check after unexpected behavior.
constexpr uint32_t STATUS_ERR_ILLEGAL   = (1 << 9);

// =============================================================================
// DMA CONTROL REGISTER BIT DEFINITIONS (DMA_CTRL @ 0x9C, ACT_DMA_CTRL @ 0xA8)
// =============================================================================

/// DMA_CTRL_START: Trigger DMA transfer
constexpr uint32_t DMA_CTRL_START   = (1 << 0);

/// DMA_CTRL_BUSY: DMA transfer in progress
constexpr uint32_t DMA_CTRL_BUSY    = (1 << 1);

/// DMA_CTRL_DONE: DMA transfer complete (write 1 to clear)
constexpr uint32_t DMA_CTRL_DONE    = (1 << 2);

// =============================================================================
// BUFFER SELECT REGISTER BIT DEFINITIONS (BUFF @ 0x28)
// =============================================================================

/// BUFF_WR_A: Select bank A for writes
constexpr uint32_t BUFF_WR_A        = (1 << 0);

/// BUFF_WR_B: Select bank B for writes
constexpr uint32_t BUFF_WR_B        = (1 << 1);

/// BUFF_RD_A: Currently reading from bank A (read-only)
constexpr uint32_t BUFF_RD_A        = (1 << 8);

/// BUFF_RD_B: Currently reading from bank B (read-only)
constexpr uint32_t BUFF_RD_B        = (1 << 9);

// =============================================================================
// HARDWARE CONSTANTS
// =============================================================================
// These must match RTL parameters in accel_top.sv

/// SYSTOLIC_ROWS: Number of PE rows in systolic array
/// 14×14 = 196 PEs, uses 196 DSP48 on Zynq-7020 (out of 220)
constexpr size_t SYSTOLIC_ROWS      = 14;  // Note: RTL uses 14, not 16

/// SYSTOLIC_COLS: Number of PE columns in systolic array
constexpr size_t SYSTOLIC_COLS      = 14;

/// BLOCK_SIZE: BSR block dimension (row and column)
/// Weight blocks are BLOCK_SIZE × BLOCK_SIZE (14×14 = 196 INT8 values)
constexpr size_t BLOCK_SIZE         = 14;

/// BLOCK_ELEMENTS: Elements per BSR block
/// 14 × 14 = 196 INT8 values = 196 bytes per block
constexpr size_t BLOCK_ELEMENTS     = BLOCK_SIZE * BLOCK_SIZE;  // 196

// =============================================================================
// ZYNQ-7020 MEMORY MAP
// =============================================================================
// Physical addresses for memory-mapped I/O on PYNQ-Z2

/// ACCEL_BASE_ADDR: Base address of CSR registers
/// Accessible via /dev/mem mmap. 4KB region (1024 32-bit registers).
/// Vivado: Set in Address Editor when adding IP to block design.
constexpr uint64_t ACCEL_BASE_ADDR      = 0x43C00000;

/// DDR_BASE_ADDR: Start of DDR memory (shared with PS)
constexpr uint64_t DDR_BASE_ADDR        = 0x00000000;

/// DDR_SIZE: Total DDR size (1GB on PYNQ-Z2)
constexpr uint64_t DDR_SIZE             = 0x40000000;  // 1GB

// =============================================================================
// RESERVED DDR REGIONS FOR ACCELERATOR BUFFERS
// =============================================================================
// Linux reserves lower memory for kernel. Upper regions for accelerator.
// Ensure these don't overlap with kernel memory (check devicetree reserved-memory).

/// ACT_BUFFER_BASE: Activation buffer region (64MB)
/// Store input activations here for DMA.
constexpr uint64_t ACT_BUFFER_BASE      = 0x10000000;

/// WGT_BUFFER_BASE: Weight buffer region (64MB)
/// Store quantized weights here (dense or BSR format).
constexpr uint64_t WGT_BUFFER_BASE      = 0x14000000;

/// OUT_BUFFER_BASE: Output buffer region (64MB)
/// DMA writes results here (future: output DMA not yet implemented).
constexpr uint64_t OUT_BUFFER_BASE      = 0x18000000;

/// BSR_BUFFER_BASE: BSR metadata region (64MB)
/// Store row_ptr, col_idx, and weight blocks for sparse layers.
constexpr uint64_t BSR_BUFFER_BASE      = 0x1C000000;

/// BUFFER_REGION_SIZE: Size of each buffer region (64MB)
constexpr size_t   BUFFER_REGION_SIZE   = 0x04000000;  // 64MB

} // namespace csr
} // namespace resnet_accel

#endif // CSR_MAP_HPP
