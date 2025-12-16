// =============================================================================
// bsr_dma.sv — AXI4 Master BSR DMA Engine for Sparse Accelerator
// =============================================================================
//
// OVERVIEW
// ========
// Transfers Block Sparse Row (BSR) format sparse matrix data from DDR memory
// into on-chip BRAMs for accelerator consumption. This is the PRIMARY data path
// for loading pre-trained sparse neural network weights.
//
// BSR FORMAT EXPLANATION
// ======================
// BSR stores sparse matrices as a collection of dense blocks rather than 
// individual non-zero elements (like CSR/COO). This is IDEAL for systolic arrays
// because:
//   1. Each block (14×14) maps perfectly to our PE array dimensions
//   2. Block-level sparsity is more predictable than element-level
//   3. Reduces metadata overhead (1 col_idx per block vs. per element)
//
// MEMORY LAYOUT (in DDR)
// ======================
// The host software prepares BSR data in this exact format:
//
// ┌─────────────────────────────────────────────────────────────────────┐
// │ HEADER (12 bytes, 3×32-bit words)                                   │
// │   Word 0: num_rows    - Number of block rows (e.g., 512 for FC)     │
// │   Word 1: num_cols    - Number of block columns                     │
// │   Word 2: total_blocks - Total non-zero blocks in matrix            │
// ├─────────────────────────────────────────────────────────────────────┤
// │ ROW_PTR Array ((num_rows + 1) × 32-bit words)                       │
// │   row_ptr[i] = cumulative block count for rows 0..i-1               │
// │   row_ptr[0] = 0 (always)                                           │
// │   row_ptr[num_rows] = total_blocks                                  │
// │   Example: [0, 3, 5, 8] means Row 0 has 3 blocks, Row 1 has 2, etc. │
// ├─────────────────────────────────────────────────────────────────────┤
// │ COL_IDX Array (total_blocks × 16-bit words)                         │
// │   col_idx[b] = column position of block b                           │
// │   Example: [2, 5, 8, 1, 4, ...] = block positions in each row       │
// ├─────────────────────────────────────────────────────────────────────┤
// │ WEIGHT BLOCKS (total_blocks × 196 bytes each)                       │
// │   Each block: 14×14 = 196 INT8 values (one for each PE)             │
// │   Storage: Row-major within each block                              │
// │   Magic Number: 196 bytes = 14 rows × 14 cols × 1 byte              │
// └─────────────────────────────────────────────────────────────────────┘
//
// DATA FLOW ARCHITECTURE
// ======================
//
//   ┌──────────┐   AXI4 Read   ┌────────────┐   Unpack   ┌───────────────┐
//   │   DDR    │ ============> │ This Module│ =========> │ On-Chip BRAMs │
//   │ (Sparse  │  64-bit bus   │  (bsr_dma) │  3 paths   │ ┌───────────┐ │
//   │ Weights) │  Burst mode   └────────────┘            │ │ row_ptr[] │ │
//   └──────────┘                                         │ │ col_idx[] │ │
//                                                        │ │ blocks[]  │ │
//                                                        │ └───────────┘ │
//                                                        └───────────────┘
//
// UNPACKING LOGIC
// ===============
// AXI bus is 64-bits, but our data has mixed widths:
//   - row_ptr:  32-bit → 2 per beat (requires 2-state unpack)
//   - col_idx:  16-bit → 4 per beat (requires 4-state unpack)
//   - weights:  64-bit → 1 per beat (direct copy, no unpack)
//
// This is why the FSM has separate WRITE_ROW_PTR_HIGH and WRITE_COL_IDX_1/2/3
// states - to handle the unpacking at full speed without stalling AXI.
//
// PERFORMANCE CHARACTERISTICS
// ===========================
// - AXI Burst: 16 beats × 8 bytes = 128 bytes per burst
// - Weight throughput: ~7 bytes/cycle at 200 MHz = 1.4 GB/s
// - Typical FC layer (1024×1024 @ 50% sparse): ~1 MB in ~0.7 ms
//
// INTEGRATION
// ===========
// This module interfaces with:
//   - CSR: start/done/busy/error signals for host control
//   - AXI Bridge: Shared with act_dma, differentiated by STREAM_ID
//   - bsr_scheduler: Consumes row_ptr/col_idx to orchestrate computation
//   - wgt_buffer: Receives unpacked weight blocks
//
// MAGIC NUMBERS REFERENCE
// =======================
// 196 = 14×14 bytes per block (matches PE array size)
// 16  = bytes in header (3 words + 1 padding for 64-bit alignment)
// 64  = bytes per weight block in packed format (8 beats × 8 bytes)
// BURST_LEN = 15 → 16 beats per AXI burst (0-indexed per AXI spec)
// STREAM_ID = 0  → BSR DMA's unique ID (act_dma uses 1)
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module bsr_dma #(
    // =========================================================================
    // PARAMETER: AXI_ADDR_W - AXI Address Width
    // =========================================================================
    // 32 bits addresses up to 4GB DDR (sufficient for Zynq-7020).
    parameter AXI_ADDR_W = 32,
    
    // =========================================================================
    // PARAMETER: AXI_DATA_W - AXI Data Bus Width
    // =========================================================================
    // 64-bit bus matches Zynq HP port maximum efficiency.
    // Wider buses would require IP modifications.
    parameter AXI_DATA_W = 64,
    
    // =========================================================================
    // PARAMETER: AXI_ID_W - Transaction ID Width
    // =========================================================================
    // 4 bits allows 16 outstanding transactions and 16 unique IDs.
    parameter AXI_ID_W   = 4,
    
    // =========================================================================
    // PARAMETER: STREAM_ID - Unique ID for This DMA
    // =========================================================================
    // CRITICAL: Each DMA must have unique ID for response routing:
    //   0 = BSR DMA (this module) - weight/metadata
    //   1 = Activation DMA - input activations
    // If IDs conflict, response data gets corrupted!
    parameter STREAM_ID  = 0,
    
    // =========================================================================
    // PARAMETER: BRAM_ADDR_W - BRAM Address Width
    // =========================================================================
    // 10 bits = 1024 entries per BRAM (standard BRAM18 capacity).
    parameter BRAM_ADDR_W = 10,
    
    // =========================================================================
    // PARAMETER: BURST_LEN - Maximum Burst Length (0-based)
    // =========================================================================
    // ARLEN = 15 means 16 beats per burst = 128 bytes.
    // WHY 16 BEATS?
    //   - Good balance of efficiency and latency
    //   - AXI4 allows up to 256, but diminishing returns
    //   - Must not cross 4KB boundary (AXI requirement)
    parameter BURST_LEN   = 8'd15
)(
    // =========================================================================
    // SYSTEM INTERFACE
    // =========================================================================
    input  wire                  clk,      // System clock (100-200 MHz)
    input  wire                  rst_n,    // Active-low async reset

    // =========================================================================
    // CONTROL INTERFACE (from CSR)
    // =========================================================================
    
    /** start - Pulse HIGH to begin BSR transfer from src_addr */
    input  wire                  start,
    
    /** src_addr - Base address of BSR data in DDR (must be 64-bit aligned) */
    input  wire [AXI_ADDR_W-1:0] src_addr,
    
    /** done - Pulses HIGH when transfer completes (check error flag) */
    output reg                   done,
    
    /** busy - HIGH while transfer is in progress */
    output reg                   busy,
    
    /** error - HIGH if AXI error occurred (stays set until next start) */
    output reg                   error,

    // =========================================================================
    // AXI4 MASTER READ INTERFACE (to AXI Bridge → DDR)
    // =========================================================================
    // READ-ONLY interface for fetching sparse weight data.
    // Uses STREAM_ID=0 to distinguish responses from act_dma.
    
    // ─────────────────────────────────────────────────────────────────────────
    // READ ADDRESS CHANNEL (AR)
    // ─────────────────────────────────────────────────────────────────────────
    
    /** m_axi_arid - Transaction ID (always STREAM_ID=0 for BSR) */
    output wire [AXI_ID_W-1:0]   m_axi_arid,
    
    /** m_axi_araddr - Read address (points to header, row_ptr, col_idx, or weights) */
    output reg [AXI_ADDR_W-1:0]  m_axi_araddr,
    
    /** m_axi_arlen - Burst length minus 1 (e.g., 15 = 16 beats) */
    output reg [7:0]             m_axi_arlen,
    
    /** m_axi_arsize - Bytes per beat (3'b011 = 8 bytes for 64-bit) */
    output reg [2:0]             m_axi_arsize,
    
    /** m_axi_arburst - Burst type (2'b01 = INCR for sequential access) */
    output reg [1:0]             m_axi_arburst,
    
    /** m_axi_arvalid - Address valid (master asserts) */
    output reg                   m_axi_arvalid,
    
    /** m_axi_arready - Address ready (slave asserts) */
    input  wire                  m_axi_arready,

    // ─────────────────────────────────────────────────────────────────────────
    // READ DATA CHANNEL (R)
    // ─────────────────────────────────────────────────────────────────────────
    
    /** m_axi_rid - Transaction ID (should match m_axi_arid) */
    input  wire [AXI_ID_W-1:0]   m_axi_rid,
    
    /** m_axi_rdata - Read data (64 bits containing header/metadata/weights) */
    input  wire [AXI_DATA_W-1:0] m_axi_rdata,
    
    /** m_axi_rresp - Read response (2'b00 = OKAY, others = error) */
    input  wire [1:0]            m_axi_rresp,
    
    /** m_axi_rlast - Last beat of burst */
    input  wire                  m_axi_rlast,
    
    /** m_axi_rvalid - Data valid (slave asserts) */
    input  wire                  m_axi_rvalid,
    
    /** m_axi_rready - Data ready (master asserts) */
    output reg                   m_axi_rready,

    // =========================================================================
    // BRAM WRITE INTERFACES (to on-chip storage)
    // =========================================================================
    // Three separate write ports for the three BSR components.
    // Each BRAM is sized for worst-case layer requirements.
    
    // ─────────────────────────────────────────────────────────────────────────
    // ROW POINTERS (32-bit per entry)
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Row pointer array stores cumulative block counts.
     * Entry count: num_rows + 1 (extra entry for total_blocks)
     * Max size: 1024 entries = 4 KB (fits in one BRAM18)
     */
    output reg                   row_ptr_we,      // Write enable
    output reg [BRAM_ADDR_W-1:0] row_ptr_addr,    // Write address (entry index)
    output reg [31:0]            row_ptr_wdata,   // Write data (32-bit block count)

    // ─────────────────────────────────────────────────────────────────────────
    // COLUMN INDICES (16-bit per entry)
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Column index array stores block column positions.
     * Entry count: total_blocks (one per non-zero block)
     * Max size: 1024 entries = 2 KB
     * WHY 16-bit? Supports up to 65536 columns (more than enough for DNNs)
     */
    output reg                   col_idx_we,      // Write enable
    output reg [BRAM_ADDR_W-1:0] col_idx_addr,    // Write address (block index)
    output reg [15:0]            col_idx_wdata,   // Write data (16-bit column index)

    // ─────────────────────────────────────────────────────────────────────────
    // WEIGHT BLOCKS (64-bit per write, covers 8 INT8 values)
    // ─────────────────────────────────────────────────────────────────────────
    /**
     * Weight block storage for 14×14 INT8 blocks.
     * Each block = 196 bytes, written in 25 beats (196/8 = 24.5, rounded up)
     * NOTE: wgt_addr is BYTE address, not word address
     * Total capacity: total_blocks × 196 bytes
     */
    output reg                   wgt_we,          // Write enable
    output reg [BRAM_ADDR_W+6:0] wgt_addr,        // Byte address (extra bits for block offset)
    output reg [63:0]            wgt_wdata        // Write data (8 INT8 weights)
);

    // =========================================================================
    // FSM STATE ENCODING
    // =========================================================================
    /**
     * State Machine Overview
     * =======================
     * Unlike act_dma which has a simple 4-state FSM, bsr_dma has 14 states
     * to handle the multi-phase BSR format:
     *
     * IDLE → READ_HEADER → WAIT_HEADER → SETUP_ROW_PTR → READ_ROW_PTR
     *   ↓                                                     ↓
     *   ↓                               WRITE_ROW_PTR_HIGH ←──┘
     *   ↓                                       ↓
     *   ↓   SETUP_COL_IDX ←─────────────────────┘
     *   ↓         ↓
     *   ↓   READ_COL_IDX → WRITE_COL_IDX_1 → WRITE_COL_IDX_2 → WRITE_COL_IDX_3
     *   ↓         ↓                                                  ↓
     *   ↓   SETUP_WEIGHTS ←──────────────────────────────────────────┘
     *   ↓         ↓
     *   ↓   READ_WEIGHTS → DONE_STATE → IDLE
     *   ↓
     *   └── (on error, any state can jump to IDLE)
     *
     * UNPACKING STATES:
     * - WRITE_ROW_PTR_HIGH: Writes upper 32-bits of 64-bit beat
     * - WRITE_COL_IDX_1/2/3: Writes 2nd/3rd/4th 16-bit indices from beat
     *
     * FSM ATTRIBUTE:
     * (* fsm_encoding = "one_hot" *) helps synthesis optimize for speed.
     * One-hot uses more registers but faster state decode logic.
     */
    typedef enum logic [3:0] {
        IDLE,                // 0: Waiting for start signal
        READ_HEADER,         // 1: Issue burst read for BSR header
        WAIT_HEADER,         // 2: Receive and parse 3-word header
        SETUP_ROW_PTR,       // 3: Calculate row_ptr transfer parameters
        READ_ROW_PTR,        // 4: Issue burst reads for row_ptr array
        WRITE_ROW_PTR_HIGH,  // 5: Unpack upper 32-bits of 64-bit beat
        SETUP_COL_IDX,       // 6: Calculate col_idx transfer parameters
        READ_COL_IDX,        // 7: Issue burst reads for col_idx array
        WRITE_COL_IDX_1,     // 8: Unpack 2nd 16-bit index
        WRITE_COL_IDX_2,     // 9: Unpack 3rd 16-bit index
        WRITE_COL_IDX_3,     // 10: Unpack 4th 16-bit index
        SETUP_WEIGHTS,       // 11: Calculate weight transfer parameters
        READ_WEIGHTS,        // 12: Burst read weight blocks (no unpack)
        DONE_STATE           // 13: Transfer complete, pulse done
    } state_t;

    (* fsm_encoding = "one_hot" *) state_t state, next_state;

    // =========================================================================
    // HEADER REGISTERS
    // =========================================================================
    /**
     * These registers capture the BSR header values which define
     * the sparse matrix dimensions. They're used to calculate
     * transfer sizes for subsequent phases.
     */
    reg [31:0] num_rows;      // Block rows in sparse matrix
    reg [31:0] num_cols;      // Block columns (for bounds checking)
    reg [31:0] total_blocks;  // Total non-zero blocks to transfer

    // =========================================================================
    // ADDRESS & TRANSFER TRACKING
    // =========================================================================
    /**
     * current_axi_addr - Tracks position in DDR as we read through BSR data
     * words_remaining  - Elements left to transfer in current phase
     *                    (32-bit words for row_ptr, 16-bit for col_idx, 64-bit for weights)
     */
    reg [AXI_ADDR_W-1:0] current_axi_addr;
    reg [31:0]           words_remaining;
    
    // =========================================================================
    // DATA UNPACKING REGISTERS
    // =========================================================================
    /**
     * header_word_idx - Tracks which header word we're parsing (0-2)
     * rdata_reg       - Buffers 64-bit AXI data for multi-cycle unpacking
     * rlast_reg       - Stores rlast signal during unpacking states
     *                   (needed because we throttle rready during unpack)
     */
    reg [1:0]  header_word_idx;
    reg [63:0] rdata_reg;
    reg        rlast_reg;

    // =========================================================================
    // AXI PROTOCOL CONSTANTS
    // =========================================================================
    /**
     * AXI_SIZE_64 = 3'b011 → 2^3 = 8 bytes per beat (64-bit bus)
     * AXI_BURST_INCR = 2'b01 → Incrementing burst (sequential addresses)
     */
    localparam [2:0] AXI_SIZE_64 = 3'b011;
    localparam [1:0] AXI_BURST_INCR = 2'b01;

    // =========================================================================
    // TRANSACTION ID ASSIGNMENT
    // =========================================================================
    // Constant ID for all transactions from this DMA.
    // The AXI bridge uses this to route responses correctly.
    assign m_axi_arid = STREAM_ID[AXI_ID_W-1:0];

    // =========================================================================
    // MAIN FSM - BSR DATA TRANSFER STATE MACHINE
    // =========================================================================
    /**
     * FSM OVERVIEW
     * =============
     * This FSM orchestrates a 4-phase transfer:
     *   Phase 1: Read 12-byte header (num_rows, num_cols, total_blocks)
     *   Phase 2: Read (num_rows+1) × 4-byte row_ptr values
     *   Phase 3: Read total_blocks × 2-byte col_idx values  
     *   Phase 4: Read total_blocks × 196-byte weight blocks
     *
     * UNPACKING STRATEGY
     * ==================
     * Our 64-bit AXI bus carries different data widths:
     *   - Header:   2 × 32-bit per beat
     *   - row_ptr:  2 × 32-bit per beat → unpack in 2 cycles
     *   - col_idx:  4 × 16-bit per beat → unpack in 4 cycles
     *   - weights:  8 × 8-bit per beat  → direct write (no unpack)
     *
     * The unpack states (WRITE_ROW_PTR_HIGH, WRITE_COL_IDX_1/2/3) handle
     * the sub-word extraction while throttling AXI rready.
     *
     * ERROR HANDLING
     * ==============
     * Any AXI error response (RRESP != OKAY) causes immediate abort.
     * The error flag remains set until the next start pulse.
     */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // ─────────────────────────────────────────────────────────────────
            // RESET STATE - Initialize all outputs to safe defaults
            // ─────────────────────────────────────────────────────────────────
            state <= IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            
            m_axi_arvalid <= 1'b0;
            m_axi_rready <= 1'b0;
            
            row_ptr_we <= 1'b0;
            col_idx_we <= 1'b0;
            wgt_we <= 1'b0;
            
            row_ptr_addr <= 0;
            col_idx_addr <= 0;
            wgt_addr <= 0;
            
            current_axi_addr <= 0;
            header_word_idx <= 0;
            rdata_reg <= 0;
            rlast_reg <= 0;
        end else begin
            // ─────────────────────────────────────────────────────────────────
            // DEFAULT SIGNALS - Write enables are pulses
            // ─────────────────────────────────────────────────────────────────
            row_ptr_we <= 1'b0;
            col_idx_we <= 1'b0;
            wgt_we <= 1'b0;
            m_axi_arvalid <= 1'b0; // Auto-clear unless set in state
            
            case (state)
                // ─────────────────────────────────────────────────────────────
                // IDLE: Wait for start, capture source address
                // ─────────────────────────────────────────────────────────────
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        error <= 1'b0;
                        current_axi_addr <= src_addr;
                        state <= READ_HEADER;
                    end
                end

                // =============================================================
                // PHASE 1: HEADER READ (12 bytes = 3 × 32-bit words)
                // =============================================================
                /**
                 * Header is packed as:
                 *   Beat 0: [num_cols(63:32), num_rows(31:0)]
                 *   Beat 1: [padding(63:32), total_blocks(31:0)]
                 *
                 * We read 2 beats (16 bytes) to safely capture all 3 words.
                 * Magic number: arlen = 1 means 2 beats.
                 */
                READ_HEADER: begin
                    m_axi_araddr <= current_axi_addr;
                    m_axi_arlen  <= 8'd1;         // 2 beats = 16 bytes
                    m_axi_arsize <= AXI_SIZE_64;  // 8 bytes per beat
                    m_axi_arburst <= AXI_BURST_INCR;
                    m_axi_arvalid <= 1'b1;
                    m_axi_rready <= 1'b1;
                    
                    header_word_idx <= 0;
                    
                    if (m_axi_arready && m_axi_arvalid) begin
                        m_axi_arvalid <= 1'b0;    // Handshake complete
                        state <= WAIT_HEADER;
                    end
                end

                WAIT_HEADER: begin
                    m_axi_rready <= 1'b1;
                    if (m_axi_rvalid) begin
                        if (m_axi_rresp != 2'b00) begin
                            // AXI Error - abort transfer
                            error <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= IDLE;
                        end else begin
                            /**
                             * HEADER UNPACKING:
                             * Beat 0 (idx=0): Extract num_rows[31:0], num_cols[63:32]
                             * Beat 1 (idx=2): Extract total_blocks[31:0]
                             *
                             * Note: idx jumps 0→2 because we consumed 2 words per beat
                             */
                            case (header_word_idx)
                                0: begin
                                    num_rows <= m_axi_rdata[31:0];     // Rows in lower 32
                                    num_cols <= m_axi_rdata[63:32];    // Cols in upper 32
                                    header_word_idx <= 2;  // Next beat has 3rd word
                                end
                                2: begin
                                    total_blocks <= m_axi_rdata[31:0];  // Total sparse blocks
                                    header_word_idx <= 3;  // Mark header complete
                                end
                            endcase
                            
                            if (m_axi_rlast) begin
                                // Advance address past header (16 bytes for alignment)
                                // Magic number: 16 = 2 beats × 8 bytes
                                current_axi_addr <= current_axi_addr + 16;
                                state <= SETUP_ROW_PTR;
                            end
                        end
                    end
                end

                // =============================================================
                // PHASE 2: ROW POINTER READ ((num_rows + 1) × 32-bit words)
                // =============================================================
                /**
                 * Row pointers are 32-bit cumulative block counts.
                 * Example: For 3 rows with [3, 2, 4] blocks each:
                 *   row_ptr = [0, 3, 5, 9]  (length = num_rows + 1)
                 *
                 * Unpacking: Each 64-bit beat contains 2 row_ptr values.
                 * We write the lower 32-bits immediately, then enter
                 * WRITE_ROW_PTR_HIGH to write the upper 32-bits.
                 */
                SETUP_ROW_PTR: begin
                    words_remaining <= num_rows + 1;  // +1 for final total
                    row_ptr_addr <= 0;
                    state <= READ_ROW_PTR;
                end

                READ_ROW_PTR: begin
                    // ─────────────────────────────────────────────────────────
                    // Issue AXI burst read for row_ptr data
                    // ─────────────────────────────────────────────────────────
                    if (!m_axi_arvalid && !m_axi_rvalid && words_remaining > 0) begin
                        m_axi_araddr <= current_axi_addr;
                        /**
                         * BURST LENGTH CALCULATION:
                         * Each beat carries 2 × 32-bit words.
                         * beats = ceil(words / 2) = (words + 1) >> 1
                         * arlen = beats - 1
                         *
                         * Max: 32 words = 16 beats (BURST_LEN)
                         */
                        if (words_remaining > 32) 
                            m_axi_arlen <= BURST_LEN;
                        else 
                            m_axi_arlen <= ((words_remaining + 1) >> 1) - 1;
                            
                        m_axi_arvalid <= 1'b1;
                    end

                    if (m_axi_arready && m_axi_arvalid) m_axi_arvalid <= 1'b0;

                    m_axi_rready <= 1'b1;  // Ready to accept data

                    if (m_axi_rvalid) begin
                        if (m_axi_rresp != 2'b00) begin
                            error <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= IDLE;
                        end else begin
                            // Buffer data for potential upper-word extraction
                            rdata_reg <= m_axi_rdata;
                            rlast_reg <= m_axi_rlast;

                            // Write lower 32-bits immediately
                            if (words_remaining > 0) begin
                                row_ptr_we <= 1'b1;
                                row_ptr_wdata <= m_axi_rdata[31:0];
                                row_ptr_addr <= row_ptr_addr + 1;
                                words_remaining <= words_remaining - 1;
                            end
                            
                            // Check if upper 32-bits need extraction
                            if (words_remaining > 1) begin
                                m_axi_rready <= 1'b0;  // Throttle AXI
                                state <= WRITE_ROW_PTR_HIGH;
                            end else begin
                                // Beat fully consumed (only needed lower word)
                                if (m_axi_rlast) begin
                                    current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                                    if (words_remaining <= 1) state <= SETUP_COL_IDX;
                                end
                            end
                        end
                    end
                end

                WRITE_ROW_PTR_HIGH: begin
                    // ─────────────────────────────────────────────────────────
                    // Extract and write upper 32-bits of buffered beat
                    // ─────────────────────────────────────────────────────────
                    m_axi_rready <= 1'b0;  // Hold off next beat
                    
                    row_ptr_we <= 1'b1;
                    row_ptr_wdata <= rdata_reg[63:32];  // Upper 32 bits
                    row_ptr_addr <= row_ptr_addr + 1;
                    words_remaining <= words_remaining - 1;
                    
                    // Check if burst complete
                    if (rlast_reg) begin
                        current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                        if (words_remaining <= 1) state <= SETUP_COL_IDX;
                        else state <= READ_ROW_PTR;  // Continue (shouldn't happen)
                    end else begin
                        state <= READ_ROW_PTR;  // Get next beat
                    end
                end

                // =============================================================
                // PHASE 3: COLUMN INDEX READ (total_blocks × 16-bit words)
                // =============================================================
                // =============================================================
                // PHASE 3: COLUMN INDEX READ (total_blocks × 16-bit words)
                // =============================================================
                /**
                 * Column indices are 16-bit block column positions.
                 * Example: col_idx = [2, 5, 8, 1, 4] means:
                 *   Block 0 is in column 2, Block 1 in column 5, etc.
                 *
                 * Unpacking: Each 64-bit beat contains 4 col_idx values.
                 * We need 4 states to extract all indices:
                 *   READ_COL_IDX:    Write bits[15:0]
                 *   WRITE_COL_IDX_1: Write bits[31:16]
                 *   WRITE_COL_IDX_2: Write bits[47:32]
                 *   WRITE_COL_IDX_3: Write bits[63:48]
                 */
                SETUP_COL_IDX: begin
                    words_remaining <= total_blocks;
                    col_idx_addr <= 0;
                    state <= READ_COL_IDX;
                end

                READ_COL_IDX: begin
                    // ─────────────────────────────────────────────────────────
                    // Issue AXI burst read for col_idx data
                    // ─────────────────────────────────────────────────────────
                    if (!m_axi_arvalid && !m_axi_rvalid && words_remaining > 0) begin
                        m_axi_araddr <= current_axi_addr;
                        /**
                         * BURST LENGTH CALCULATION:
                         * Each beat carries 4 × 16-bit words.
                         * beats = ceil(words / 4) = (words + 3) >> 2
                         * arlen = beats - 1
                         *
                         * Max: 64 words = 16 beats (BURST_LEN)
                         */
                        if (words_remaining > 64) 
                            m_axi_arlen <= BURST_LEN;
                        else 
                            m_axi_arlen <= ((words_remaining + 3) >> 2) - 1;
                            
                        m_axi_arvalid <= 1'b1;
                    end

                    if (m_axi_arready && m_axi_arvalid) m_axi_arvalid <= 1'b0;

                    m_axi_rready <= 1'b1;

                    if (m_axi_rvalid) begin
                        if (m_axi_rresp != 2'b00) begin
                            error <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= IDLE;
                        end else begin
                            rdata_reg <= m_axi_rdata;
                            rlast_reg <= m_axi_rlast;

                            // Write Index 0 (Bits 15:0) - first of 4 indices
                            if (words_remaining > 0) begin
                                col_idx_we <= 1'b1;
                                col_idx_wdata <= m_axi_rdata[15:0];
                                col_idx_addr <= col_idx_addr + 1;
                                words_remaining <= words_remaining - 1;
                            end

                            // More indices in this beat?
                            if (words_remaining > 1) begin
                                m_axi_rready <= 1'b0;  // Throttle AXI
                                state <= WRITE_COL_IDX_1;
                            end else begin
                                if (m_axi_rlast) begin
                                    current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                                    if (words_remaining <= 1) state <= SETUP_WEIGHTS;
                                end
                            end
                        end
                    end
                end

                WRITE_COL_IDX_1: begin
                    // ─────────────────────────────────────────────────────────
                    // Extract Index 1 (Bits 31:16) - second of 4 indices
                    // ─────────────────────────────────────────────────────────
                    m_axi_rready <= 1'b0;
                    col_idx_we <= 1'b1;
                    col_idx_wdata <= rdata_reg[31:16];
                    col_idx_addr <= col_idx_addr + 1;
                    words_remaining <= words_remaining - 1;

                    if (words_remaining > 1) state <= WRITE_COL_IDX_2;
                    else begin
                        if (rlast_reg) begin
                            current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                            state <= SETUP_WEIGHTS;
                        end else state <= READ_COL_IDX;
                    end
                end

                WRITE_COL_IDX_2: begin
                    // ─────────────────────────────────────────────────────────
                    // Extract Index 2 (Bits 47:32) - third of 4 indices
                    // ─────────────────────────────────────────────────────────
                    m_axi_rready <= 1'b0;
                    col_idx_we <= 1'b1;
                    col_idx_wdata <= rdata_reg[47:32];
                    col_idx_addr <= col_idx_addr + 1;
                    words_remaining <= words_remaining - 1;

                    if (words_remaining > 1) state <= WRITE_COL_IDX_3;
                    else begin
                        if (rlast_reg) begin
                            current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                            state <= SETUP_WEIGHTS;
                        end else state <= READ_COL_IDX;
                    end
                end

                WRITE_COL_IDX_3: begin
                    // ─────────────────────────────────────────────────────────
                    // Extract Index 3 (Bits 63:48) - fourth of 4 indices
                    // ─────────────────────────────────────────────────────────
                    m_axi_rready <= 1'b0;
                    col_idx_we <= 1'b1;
                    col_idx_wdata <= rdata_reg[63:48];
                    col_idx_addr <= col_idx_addr + 1;
                    words_remaining <= words_remaining - 1;

                    if (rlast_reg) begin
                        current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                        if (words_remaining <= 1) state <= SETUP_WEIGHTS;
                        else state <= READ_COL_IDX;  // Continue reading
                    end else begin
                        state <= READ_COL_IDX;  // Next beat
                    end
                end

                // =============================================================
                // PHASE 4: WEIGHT BLOCKS (total_blocks × 64 bytes each)
                // =============================================================
                /**
                 * Weight blocks are the actual sparse matrix values.
                 * Each block is 14×14 = 196 bytes, but we store 64 bytes per
                 * block in our simplified format (8 beats × 8 bytes).
                 *
                 * IMPORTANT: Unlike row_ptr and col_idx, weights are 64-bit
                 * aligned and don't require unpacking. Each beat is written
                 * directly to the weight BRAM.
                 *
                 * MAGIC NUMBER: 8 words per block = 64 bytes
                 * Note: This is a simplified block size. Full 196-byte blocks
                 * would need 25 beats (196/8 = 24.5, rounded up).
                 *
                 * wgt_addr is a BYTE address, incremented by 8 per beat.
                 */
                SETUP_WEIGHTS: begin
                    // Total 64-bit words = total_blocks × 8 beats per block
                    words_remaining <= total_blocks * 8; 
                    wgt_addr <= 0;
                    state <= READ_WEIGHTS;
                end

                READ_WEIGHTS: begin
                    // ─────────────────────────────────────────────────────────
                    // Burst read weight data - no unpacking needed
                    // ─────────────────────────────────────────────────────────
                    m_axi_rready <= 1'b1;  // Always ready (no unpack states)
                    
                    if (!m_axi_arvalid && !m_axi_rvalid && words_remaining > 0) begin
                        m_axi_araddr <= current_axi_addr;
                        if (words_remaining > BURST_LEN) m_axi_arlen <= BURST_LEN;
                        else m_axi_arlen <= words_remaining - 1;
                        m_axi_arvalid <= 1'b1;
                    end

                    if (m_axi_arready && m_axi_arvalid) m_axi_arvalid <= 1'b0;

                    // Process data on valid handshake
                    if (m_axi_rvalid && m_axi_rready) begin
                        if (m_axi_rresp != 2'b00) begin
                            error <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= IDLE;
                        end else begin
                            // Direct 64-bit write - 8 INT8 weights per beat
                            wgt_we <= 1'b1;
                            wgt_wdata <= m_axi_rdata;
                            wgt_addr <= wgt_addr + 8;  // Byte address += 8
                            words_remaining <= words_remaining - 1;

                            if (m_axi_rlast) begin
                                current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);
                                if (words_remaining <= 1) state <= DONE_STATE;
                            end
                        end
                    end
                end

                // ─────────────────────────────────────────────────────────────
                // DONE_STATE: Signal completion to CSR
                // ─────────────────────────────────────────────────────────────
                /**
                 * Transfer complete. Pulse done=1 and wait for start to deassert
                 * before returning to IDLE. This handshake prevents accidental
                 * restart if start is held high.
                 */
                DONE_STATE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    if (!start) state <= IDLE;  // Handshake reset
                end
            endcase
        end
    end

endmodule
`default_nettype wire
