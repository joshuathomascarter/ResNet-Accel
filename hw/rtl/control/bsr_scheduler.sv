// =============================================================================
// bsr_scheduler.sv — Block-Sparse Row (BSR) Scheduler for Sparse Inference
// =============================================================================
//
// OVERVIEW
// ========
// This scheduler orchestrates sparse matrix computation using the BSR (Block
// Sparse Row) format. Unlike the dense scheduler, it only processes non-zero
// blocks, achieving significant speedup for sparse neural network weights.
//
// SPARSE VS DENSE COMPARISON
// ==========================
// Dense GEMM: O(M × N × K) operations regardless of sparsity
// BSR GEMM:   O(NNZ_blocks × M) operations where NNZ = non-zero blocks
//
// Example: ResNet-18 FC layer with 50% sparsity
//   Dense: 1024 × 1024 × 1024 = 1B operations
//   BSR:   512K blocks × 1024 = 524M operations (2× speedup)
//
// BSR FORMAT RECAP
// ================
// ┌─────────────────────────────────────────────────────────────────────┐
// │ row_ptr[k]   = Index of first block in row k                        │
// │ row_ptr[k+1] = Index of first block in row k+1                      │
// │ Block count for row k = row_ptr[k+1] - row_ptr[k]                   │
// │                                                                      │
// │ col_idx[b]   = Column position of block b                           │
// │ blocks[b]    = 14×14 weight values for block b                      │
// └─────────────────────────────────────────────────────────────────────┘
//
// SPARSE ITERATION ALGORITHM
// ==========================
// The key insight is that we iterate over K (rows of B) and only process
// the non-zero blocks in each row:
//
//   for k = 0 to KT-1:                    // Iterate K tiles
//     start = row_ptr[k]
//     end   = row_ptr[k+1]
//     for blk = start to end-1:           // Only non-zero blocks!
//       n = col_idx[blk]                  // Get column from metadata
//       Load Weight Block[blk] into PEs   // Weight-Stationary
//       for m = 0 to MT-1:                // All activation tiles
//         Stream Activation[m,k]
//         C[m,n] += A[m,k] × B[blk]       // Accumulate
//
// This skips all (k,n) pairs where B is zero, achieving sparse speedup.
//
// STATE MACHINE
// =============
//
//   ┌─────────────┐
//   │   S_IDLE    │ ← Wait for start
//   └─────┬───────┘
//         ▼
//   ┌───────────────┐     ┌───────────────┐
//   │ S_FETCH_PTR1  │────▶│ S_FETCH_PTR2  │ ← Read row_ptr[k], row_ptr[k+1]
//   └───────────────┘     └───────┬───────┘
//                                 ▼
//                         ┌───────────────┐
//                         │  S_CALC_LEN   │ ← Calculate block count
//                         └───────┬───────┘
//               Empty row?        │        Non-zero blocks?
//         ┌───────────────────────┴──────────────────────┐
//         ▼                                              ▼
//   ┌───────────┐                                 ┌─────────────┐
//   │  S_NEXT_K │ ← Skip empty row                │ S_FETCH_COL │ ← Get col_idx
//   └─────┬─────┘                                 └──────┬──────┘
//         │                                              ▼
//         │                                       ┌─────────────┐
//         │                                       │  S_LOAD_WGT │ ← Load 14 weight rows
//         │                                       └──────┬──────┘
//         │                                              ▼
//         │                                       ┌─────────────┐
//         │                                       │ S_STREAM_ACT│ ← Stream all M tiles
//         │                                       └──────┬──────┘
//         │                                              ▼
//         │                                       ┌─────────────┐
//         │◀────────────More blocks?──────────────│ S_NEXT_BLK  │
//         │                                       └─────────────┘
//         │
//         ▼ (k == KT-1)
//   ┌───────────┐
//   │   DONE    │
//   └───────────┘
//
// MAGIC NUMBERS
// =============
// BLOCK_SIZE = 14: Matches systolic array dimensions (14×14)
// LOAD_CNT_MAX = 13: Load 14 weight rows (0-13)
// meta_raddr offset 128: Column indices start at offset 128 in metadata BRAM
//
// WEIGHT-STATIONARY INTEGRATION
// =============================
// load_weight_r is pipelined by 1 cycle to match SRAM read latency.
// When load_weight=1, PEs latch the incoming weight value.
// When pe_en=1 (during S_STREAM_ACT), PEs compute using stationary weights.
//
// =============================================================================

`default_nettype none

module bsr_scheduler #(
    // =========================================================================
    // PARAMETER: M_W - M Dimension Bit Width
    // =========================================================================
    parameter M_W  = 10,
    
    // =========================================================================
    // PARAMETER: N_W - N Dimension Bit Width
    // =========================================================================
    parameter N_W  = 10,
    
    // =========================================================================
    // PARAMETER: K_W - K Dimension Bit Width
    // =========================================================================
    // K is typically larger (matrix inner dimension for GEMM)
    parameter K_W  = 12,
    
    // =========================================================================
    // PARAMETER: ADDR_W - Buffer Address Width
    // =========================================================================
    parameter ADDR_W = 32,
    
    // =========================================================================
    // PARAMETER: BLOCK_SIZE - BSR Block Dimension
    // =========================================================================
    // Must match systolic array size (14×14 for our design).
    // Each block contains BLOCK_SIZE² = 196 INT8 weights.
    parameter BLOCK_SIZE = 14
)(
    // =========================================================================
    // SYSTEM INTERFACE
    // =========================================================================
    input  wire                 clk,
    input  wire                 rst_n,

    // =========================================================================
    // CONTROL INTERFACE
    // =========================================================================
    input  wire                 start,   // Pulse to begin sparse computation
    input  wire                 abort,   // Emergency stop
    output reg                  busy,    // Computation in progress
    output reg                  done,    // Computation complete pulse

    // =========================================================================
    // CONFIGURATION
    // =========================================================================
    input  wire [M_W-1:0]       MT,      // Number of M tiles (activation rows)
    input  wire [K_W-1:0]       KT,      // Number of K tiles (BSR rows)

    // =========================================================================
    // METADATA INTERFACE (to meta_decode.sv or BRAM)
    // =========================================================================
    /**
     * Interface to BSR metadata storage (row_ptr and col_idx arrays).
     * Simple request/valid handshake protocol.
     */
    output reg  [31:0]          meta_raddr,   // Metadata read address
    output reg                  meta_ren,     // Metadata read enable
    input  wire [31:0]          meta_rdata,   // Metadata read data
    input  wire                 meta_rvalid,  // Metadata read valid
    output wire                 meta_ready,   // Always ready (no backpressure)

    // =========================================================================
    // BUFFER INTERFACES
    // =========================================================================
    output reg                  wgt_rd_en,    // Weight buffer read enable
    output reg  [ADDR_W-1:0]    wgt_addr,     // Weight buffer address
    output reg                  act_rd_en,    // Activation buffer read enable
    output reg  [ADDR_W-1:0]    act_addr,     // Activation buffer address

    // =========================================================================
    // SYSTOLIC ARRAY CONTROL
    // =========================================================================
    /**
     * load_weight: Latch incoming weight into PE registers.
     *              Pipelined by 1 cycle to match SRAM latency.
     * pe_en:       Enable MAC computation (block_valid signal).
     * accum_en:    Enable accumulator update.
     * bypass_out:  Per-PE bypass for residual connections (ResNet).
     */
    output wire                 load_weight,
    output reg                  pe_en,
    output reg                  accum_en,
    output reg [(BLOCK_SIZE*BLOCK_SIZE)-1:0] bypass_out
);

    // =========================================================================
    // FSM STATE ENCODING (One-Hot for Performance)
    // =========================================================================
    localparam [8:0] 
        S_IDLE        = 9'b000000001,  // Waiting for start
        S_FETCH_PTR1  = 9'b000000010,  // Read row_ptr[k]
        S_FETCH_PTR2  = 9'b000000100,  // Read row_ptr[k+1]
        S_CALC_LEN    = 9'b000001000,  // Calculate block count for this row
        S_FETCH_COL   = 9'b000010000,  // Read col_idx[blk] → n_tile
        S_LOAD_WGT    = 9'b000100000,  // Load 14×14 weight block into PEs
        S_STREAM_ACT  = 9'b001000000,  // Stream activations (loop over M)
        S_NEXT_BLK    = 9'b010000000,  // Advance to next block in k-row
        S_NEXT_K      = 9'b100000000;  // Advance to next k-row

    (* fsm_encoding = "one_hot"*) reg [8:0] state, state_n;

    // =========================================================================
    // INTERNAL REGISTERS
    // =========================================================================
    reg [K_W-1:0] k_idx;      // Current K tile (row of sparse matrix)
    reg [M_W-1:0] m_idx;      // Current M tile (activation row)
    reg [4:0]     load_cnt;   // Weight load counter (0 to BLOCK_SIZE-1)
    
    // Derived constant for load counter comparison
    // BLOCK_SIZE-1 = 13 for 14×14 blocks
    localparam LOAD_CNT_MAX = BLOCK_SIZE - 1;
    
    reg [31:0]    blk_ptr;    // Current block index in col_idx/blocks arrays
    reg [31:0]    blk_end;    // End index for current k-row (from row_ptr[k+1])
    reg [31:0]    n_idx;      // Decoded N tile from col_idx[blk_ptr]

    // Pipeline register for load_weight (1-cycle delay for SRAM latency)
    reg load_weight_r;
    assign load_weight = load_weight_r;

    // Metadata latch for row_ptr[k] (needed across states)
    reg [31:0] ptr_start_reg;

  // ---------------------------------------------------------------------------
  // 3. Next State Logic
  // ---------------------------------------------------------------------------
  always @(*) begin
    state_n = state;
    case (state)
      S_IDLE:       if (start) state_n = S_FETCH_PTR1;
      
      S_FETCH_PTR1: if (meta_rvalid) state_n = S_FETCH_PTR2;
      
      S_FETCH_PTR2: if (meta_rvalid) state_n = S_CALC_LEN;
      
      S_CALC_LEN:   state_n = (ptr_start_reg == meta_rdata) ? S_NEXT_K : S_FETCH_COL; 
                    // If ptr_start == ptr_end, row is empty -> Next K
      
      S_FETCH_COL:  if (meta_rvalid) state_n = S_LOAD_WGT;
      
      S_LOAD_WGT:   state_n = (load_cnt == LOAD_CNT_MAX[4:0]) ? S_STREAM_ACT : S_LOAD_WGT;
      
      S_STREAM_ACT: if (m_idx == MT - 1) state_n = S_NEXT_BLK;
                    // Stream all M tiles against this Weight Block
      
      S_NEXT_BLK:   state_n = (blk_ptr < blk_end) ? S_FETCH_COL : S_NEXT_K;
      
      S_NEXT_K:     state_n = (k_idx < KT - 1) ? S_FETCH_PTR1 : S_IDLE; // Done if k==KT-1
      
      default:      state_n = S_IDLE;
    endcase
  end

  // ---------------------------------------------------------------------------
  // 4. Datapath & Output Logic
  // ---------------------------------------------------------------------------
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_IDLE;
      k_idx <= 0;
      m_idx <= 0;
      load_cnt <= 0;
      blk_ptr <= 0;
      blk_end <= 0;
      
      // Outputs
      meta_ren <= 0;
      wgt_rd_en <= 0;
      act_rd_en <= 0;
      load_weight_r <= 0;
      pe_en <= 0;
      busy <= 0;
      done <= 0;
    end else begin
      state <= state_n;
      
      // Default Pulses
      meta_ren <= 0;
      wgt_rd_en <= 0;
      act_rd_en <= 0;
      // load_weight_r <= 0; // Don't clear here, controlled explicitly in states
      pe_en <= 0;
      done <= 0;

      case (state)
        S_IDLE: begin
          busy <= 0;
          k_idx <= 0;
          load_cnt <= 0;
          load_weight_r <= 0;
          if (start) busy <= 1;
        end

        S_FETCH_PTR1: begin
          // Read row_ptr[k]
          meta_ren <= 1;
          meta_raddr <= k_idx; 
          if (meta_rvalid) ptr_start_reg <= meta_rdata;
        end

        S_FETCH_PTR2: begin
          // Read row_ptr[k+1]
          meta_ren <= 1;
          meta_raddr <= k_idx + 1;
          // meta_rdata will be ptr_end
        end

        S_CALC_LEN: begin
          // Setup block loop
          blk_ptr <= ptr_start_reg;
          blk_end <= meta_rdata; // ptr_end from previous state
        end

        S_FETCH_COL: begin
          // Read col_idx[blk_ptr]
          meta_ren <= 1;
          meta_raddr <= 128 + blk_ptr; // Offset 128 for col_idx table
          if (meta_rvalid) n_idx <= meta_rdata;
        end

        S_LOAD_WGT: begin
          // Load Weight Block (B_k,n) - BLOCK_SIZE cycles
          wgt_rd_en <= 1;
          // Address = (Block Index * Block Size) + Row Offset
          // We use a shift if BLOCK_SIZE is power of 2, or multiply otherwise.
          // For 14x14, this is (blk_ptr * 14) which needs multiplier.
          wgt_addr <= (blk_ptr * BLOCK_SIZE) + load_cnt; 
          
          // FIX: Pipeline load_weight to match SRAM latency
          load_weight_r <= 1;    
          
          load_cnt <= load_cnt + 1;
          
          if (load_cnt == LOAD_CNT_MAX[4:0]) begin
             m_idx <= 0;
             load_cnt <= 0; 
          end
        end

        S_STREAM_ACT: begin
          // Stop loading weights (delayed by 1 cycle effectively)
          load_weight_r <= 0;

          // Stream Activation Tile (A_m,k)
          act_rd_en <= 1;
          // Address logic: A is stored row-major (M, K). 
          // We need tile at (m_idx, k_idx).
          // Assuming linear addressing: addr = m_idx * KT + k_idx
          act_addr <= m_idx * KT + k_idx; 
          
          pe_en <= 1; // Enable MAC (block_valid = 1)
          
          if (m_idx < MT - 1) m_idx <= m_idx + 1;
        end

        S_NEXT_BLK: begin
          blk_ptr <= blk_ptr + 1;
        end

        S_NEXT_K: begin
          k_idx <= k_idx + 1;
          if (k_idx == KT - 1) begin
            done <= 1;
            busy <= 0;
          end
        end
      endcase
      
      if (abort) begin
        state <= S_IDLE;
        busy <= 0;
      end
    end
  end

  // Always ready to accept metadata when requested
  assign meta_ready = 1'b1;

  // NEW: Bypass signal control (disabled by default for now)
  // Can be extended to support instruction-based control for ResNet residual layers
  always @(*) begin
    // For now, bypass is always disabled (all PEs use normal MAC mode)
    // Set to all 1s to enable residual bypass mode for a tile
    bypass_out = {(BLOCK_SIZE*BLOCK_SIZE){1'b0}};
  end

endmodule
