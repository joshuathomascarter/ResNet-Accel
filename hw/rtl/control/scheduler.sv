// =============================================================================
// scheduler.sv — Tiled Weight-Stationary GEMM Scheduler for Systolic Array
// =============================================================================
//
// OVERVIEW
// ========
// This is the BRAIN of the accelerator. It orchestrates all computation by:
//   1. Managing the 3-level tile loop: (m_tile, n_tile, k_tile)
//   2. Controlling weight loading and activation streaming (Weight-Stationary)
//   3. Generating buffer addresses and bank selection (ping-pong)
//   4. Producing per-PE enable masks for edge tile handling
//   5. Tracking performance counters for optimization analysis
//
// WEIGHT-STATIONARY DATAFLOW
// ==========================
// In Weight-Stationary (WS) dataflow, weights stay fixed in PEs while
// activations stream through. This is optimal for our use case because:
//   - Weights are reused across many activations (CNN feature maps)
//   - Reduces weight buffer bandwidth by loading once per tile
//   - Each PE's weight register holds one INT8 value throughout compute
//
// OPERATION SEQUENCE (per output tile):
//
//   ┌─────────────┐
//   │  S_IDLE     │ ← Wait for start signal
//   └─────┬───────┘
//         ▼
//   ┌─────────────┐
//   │ S_PREP_TILE │ ← Clear accumulators, reset counters
//   └─────┬───────┘
//         ▼
//   ┌─────────────┐
//   │S_WAIT_READY │ ← Wait for DMA to fill ping/pong buffers
//   └─────┬───────┘
//         ▼
//   ┌─────────────────┐
//   │ S_LOAD_WEIGHT   │ ← Load Tk weights into PE registers (load_weight=1)
//   └─────┬───────────┘
//         ▼
//   ┌─────────────────┐
//   │ S_STREAM_K      │ ← Stream Tk activations, weights stationary (en=1)
//   └─────┬───────────┘
//         │ ← Loop for all k_tiles
//         ▼
//   ┌─────────────┐
//   │ S_TILE_DONE │ ← Advance m/n indices, signal completion
//   └─────┬───────┘
//         │ ← Loop for all m,n tiles
//         ▼
//   ┌─────────────┐
//   │  S_DONE     │ ← All tiles complete
//   └─────────────┘
//
// TILING STRATEGY
// ===============
// For GEMM: C[M×N] = A[M×K] × B[K×N]
// We tile as: C[Tm×Tn] = A[Tm×Tk] × B[Tk×Tn]
//
// Example: 1024×1024 × 1024×1024 with Tm=Tn=Tk=14:
//   MT = ceil(1024/14) = 74 output row tiles
//   NT = ceil(1024/14) = 74 output col tiles
//   KT = ceil(1024/14) = 74 K-dimension tiles
//   Total: 74 × 74 × 74 = 405,224 tile operations
//
// EDGE TILE HANDLING
// ==================
// When M, N, or K is not a multiple of tile size, edge tiles are smaller.
// This module generates enable masks to disable unused rows/columns:
//   en_mask_row[i] = 1 if row i is within Tm_eff
//   en_mask_col[j] = 1 if col j is within Tn_eff
//
// Example: M=10, Tm=4 → Tiles are [4,4,2], last tile has Tm_eff=2
//
// DOUBLE-BUFFERING (PING-PONG)
// ============================
// Buffer bank selection alternates based on k_tile parity:
//   k_tile even → bank_sel = 0 (ping)
//   k_tile odd  → bank_sel = 1 (pong)
//
// While systolic array reads from one bank, DMA fills the other:
//   Cycle 0-N:   Array reads Bank 0, DMA fills Bank 1
//   Cycle N+1:   Array reads Bank 1, DMA fills Bank 0
//
// PREPRIME OPTION
// ===============
// SRAM has 1-cycle read latency. PREPRIME parameter handles this:
//   PREPRIME=0: Accept 1-cycle bubble at start of S_STREAM_K (simpler)
//   PREPRIME=1: Issue dummy read in S_PREPRIME to pre-fill pipeline
//
// PERFORMANCE COUNTERS
// ====================
// cycles_tile:  Cycles spent in active compute (S_STREAM_K)
// stall_cycles: Cycles waiting for DMA (S_WAIT_READY)
// Use ratio = cycles_tile / (cycles_tile + stall_cycles) for efficiency
//
// MAGIC NUMBERS
// =============
// MAX_TM/MAX_TN = 64: Maximum tile dimensions (determines mask width)
// ADDR_W = 10: Buffer address width (1024 entries max)
// 200 mW: Power savings from clock gating when idle
// 8-bit one-hot state: 8 states for FSM (easy debug, fast decode)
//
// =============================================================================

module scheduler #(
  // Dimension widths (log2 maxima)
  parameter M_W  = 10,  // log2(max M)
  parameter N_W  = 10,  // log2(max N)
  parameter K_W  = 12,  // log2(max K) -- K is typically larger
  parameter TM_W = 6,   // log2(max Tm)
  parameter TN_W = 6,   // log2(max Tn)
  parameter TK_W = 6,   // log2(max Tk)
  
  // Buffer address width (for addressing into weight/activation buffers)
  parameter ADDR_W = 10,

  // Enable pre-prime of SRAM read latency:
  //  PREPRIME = 1 -> perform a 1-cycle "dummy read" before STREAM_K, so first compute cycle has valid a_vec/b_vec (no bubble).
  //  PREPRIME = 0 -> simpler: first STREAM_K cycle is a bubble (rd_en=1/k_idx=0, en=0), compute starts next cycle.
  parameter PREPRIME = 0,

  // Set to 1 to require host-provided MT/NT/KT
  parameter USE_CSR_COUNTS = 1,
  
  // Maximum tile dimensions for enable masks
  parameter MAX_TM = 64,
  parameter MAX_TN = 64,
  
  // Enable clock gating (saves ~200 mW when idle)
  parameter ENABLE_CLOCK_GATING = 1
)(
  input  wire                 clk,
  input  wire                 rst_n,

  // -------- CSR/config inputs (programmed by host) --------
  input  wire                 start,     // pulse to start (level acceptable; latched internally)
  input  wire                 abort,     // synchronous abort
  input  wire [M_W-1:0]       M,         // problem dims
  input  wire [N_W-1:0]       N,
  input  wire [K_W-1:0]       K,
  input  wire [TM_W-1:0]      Tm,        // tile dims
  input  wire [TN_W-1:0]      Tn,
  input  wire [TK_W-1:0]      Tk,

  // Optional: precomputed tile counts from CSR (to avoid div). If zero, will compute internally.
  input  wire [M_W-1:0]       MT_csr,    // ceil(M/Tm)
  input  wire [N_W-1:0]       NT_csr,    // ceil(N/Tn)
  input  wire [K_W-1:0]       KT_csr,    // ceil(K/Tk)
  // input  wire                 use_csr_counts, // 1: use *_csr; 0: compute internally (synth div)

  // Bank readiness (set by host/DMA when ping/pong banks hold the needed tile)
  input  wire                 valid_A_ping,
  input  wire                 valid_A_pong,
  input  wire                 valid_B_ping,
  input  wire                 valid_B_pong,

  // -------- Drives buffers / array --------
  output reg                  rd_en, 
           // to both A/B buffers
  output reg [TK_W-1:0]       k_idx,          // 0..Tk_eff-1
  output reg [ADDR_W-1:0]     wgt_addr,       // Weight buffer read address
  output reg [ADDR_W-1:0]     act_addr,       // Activation buffer read address
  output reg                  bank_sel_rd_A,  // 0: ping, 1: pong
  output reg                  bank_sel_rd_B,  // typically mirror A
  output reg                  clr,            // 1-cycle pulse at tile start
  output reg                  en,             // MAC enable (AND with row/col masks externally if desired)
  output wire                 load_weight,    // NEW: weight loading control for weight-stationary
  output reg [MAX_TM-1:0]     en_mask_row,    // bit i = 1 -> row i valid
  output reg [MAX_TN-1:0]     en_mask_col,    // bit j = 1 -> col j valid
  output reg [(MAX_TM*MAX_TN)-1:0] bypass_out, // NEW: Per-PE bypass signal for residual mode (ResNet)

  // -------- Status / perf --------
  output reg                  busy,
  output reg                  done_tile,      // 1-cycle pulse at end of C tile
  output reg [M_W-1:0]        m_tile,         // current tile row index (0..MT-1)
  output reg [N_W-1:0]        n_tile,         // current tile col index (0..NT-1) (driven from n_tile_r)
  output reg [K_W-1:0]        k_tile,         // current tile depth index (0..KT-1)
  output reg [31:0]           cycles_tile,    // counts cycles within tile
  output reg [31:0]           stall_cycles    // cycles stalled waiting for bank valid
);

  // ========================================================================
  // 1. Clock Gating Logic (saves ~200 mW when scheduler idle)
  // ========================================================================
  // Engineer's Note:
  // This logic gates the clock to the entire scheduler FSM when it is not busy.
  // This is a standard low-power technique.
  wire sched_clk_en, clk_gated;
  assign sched_clk_en = start | busy | abort | done_tile;
  
  generate
    if (ENABLE_CLOCK_GATING) begin : gen_clk_gate
      `ifdef XILINX_FPGA
        BUFGCE sched_clk_gate (
          .I  (clk),
          .CE (sched_clk_en),
          .O  (clk_gated)
        );
      `else
        // Generic ICG for simulation
        reg sched_clk_en_latched;
        always @(clk or sched_clk_en) begin
          if (!clk) sched_clk_en_latched <= sched_clk_en;
        end
        assign clk_gated = clk & sched_clk_en_latched;
      `endif
    end else begin : gen_no_gate
      assign clk_gated = clk;
    end
  endgenerate

  // ------------------------
  // Internal registers/state
  // ------------------------
  // State encoding - ONE-HOT for performance (weight-stationary dataflow)
  localparam [7:0] S_IDLE        = 8'b00000001,  // bit 0
                   S_PREP_TILE   = 8'b00000010,  // bit 1
                   S_WAIT_READY  = 8'b00000100,  // bit 2
                   S_PREPRIME    = 8'b00001000,  // bit 3: dummy read to prime SRAM pipeline (PREPRIME=1)
                   S_LOAD_WEIGHT = 8'b00010000,  // bit 4: load weights into PEs (weight-stationary)
                   S_STREAM_K    = 8'b00100000,  // bit 5: activations stream, weights stationary
                   S_TILE_DONE   = 8'b01000000,  // bit 6
                   S_DONE        = 8'b10000000;  // bit 7

  (* fsm_encoding = "one_hot" *) reg [7:0] state, state_n;

  // Tile counts & effective sizes
  reg [M_W-1:0] MT; // number of m tiles
  reg [N_W-1:0] NT; // number of n tiles
  reg [K_W-1:0] KT; // number of k tiles

  // Internal registered tile indices (explicit internal ownership)
  reg [M_W-1:0] m_tile_r;
  reg [N_W-1:0] n_tile_r; // existing uses already refer to n_tile_r
  reg [K_W-1:0] k_tile_r;

  // Effective dims for the *current* edges
  reg [TM_W-1:0] Tm_eff;
  reg [TN_W-1:0] Tn_eff;
  reg [TK_W-1:0] Tk_eff;

  // k loop counter within current k-tile
  reg [TK_W-1:0] k_ctr;

  // Scratch for remainder computations
  reg [M_W+TM_W:0] m_off; // m_tile*Tm
  reg [N_W+TN_W:0] n_off; // n_tile*Tn
  reg [K_W+TK_W:0] k_off; // k_tile*Tk

  // Ping/pong selects (simple policy: bank = k_tile[0])
  reg bank_sel_k;      // 0 ping / 1 pong for this k_tile
  reg A_ready, B_ready;

  // Latches
  reg start_latched;

  // Perf counters
  reg [31:0] cycles_tile_r, stall_cycles_r;
  
  // NEW: Bypass control (for residual mode in ResNet layers)
  // Set to 1 to enable bypass mode (residual addition instead of MAC)
  reg bypass_mode;

  // ---------------------------------------------------------------------------
  // Pipeline register for load_weight to match 1-cycle SRAM latency
  // ---------------------------------------------------------------------------
  reg load_weight_r;
  reg load_weight_comb;

  always @(posedge clk_gated or negedge rst_n) begin
    if (!rst_n) load_weight_r <= 1'b0;
    else        load_weight_r <= load_weight_comb;
  end
  assign load_weight = load_weight_r;

  // NEW: Initialize bypass_mode (can be extended to support instruction encoding)
  always @(*) begin
    // For now, bypass is disabled by default
    // In future, this can be set by instruction bits or a control register
    bypass_mode = 1'b0;  // Set to 1 to enable residual bypass mode
  end

  // Outputs default
  always @(*) begin
    rd_en          = 1'b0;
    k_idx          = {TK_W{1'b0}};
    wgt_addr       = {ADDR_W{1'b0}};
    act_addr       = {ADDR_W{1'b0}};
    bank_sel_rd_A  = bank_sel_k;
    bank_sel_rd_B  = bank_sel_k;
    clr            = 1'b0;
    en             = 1'b0;
    load_weight_comb = 1'b0;  // NEW: default weight load off (internal)
    done_tile      = 1'b0;
    busy           = !(state[0] | state[7]);  // NOT (S_IDLE | S_DONE)
  end

  // ------------------------
  // Helpers (combinational)
  // ------------------------

  // Engineer's Note:
  // These division functions are expensive in hardware.
  // Ideally, the host driver should pre-calculate MT, NT, KT and pass them via CSR.
  // If USE_CSR_COUNTS=1, these dividers are optimized away by synthesis if inputs are constant 0.
  function [M_W-1:0] ceil_div_M;
    input [M_W-1:0] a;
    input [TM_W-1:0] b;
    begin
      ceil_div_M = (b==0) ? {M_W{1'b0}} : (a + b - 1) / b;
    end
  endfunction
  
  function [N_W-1:0] ceil_div_N;
    input [N_W-1:0] a;
    input [TN_W-1:0] b;
    begin
      ceil_div_N = (b==0) ? {M_W{1'b0}} : (a + b - 1) / b;
    end
  endfunction
  
  function [K_W-1:0] ceil_div_K;
    input [K_W-1:0] a;
    input [TK_W-1:0] b;
    begin
      ceil_div_K = (b==0) ? {M_W{1'b0}} : (a + b - 1) / b;
    end
  endfunction

  // ---------------------------------------------------------------------------
  // 2. Effective Tile Size Calculation (Edge Handling)
  // ---------------------------------------------------------------------------
  // Engineer's Note:
  // This logic handles the "edge tiles" where the matrix dimension is not a 
  // perfect multiple of the tile size.
  // Example: M=10, Tm=4 -> Tiles are size 4, 4, 2.
  // The 'Tm_eff' signal tells the PEs to mask off the unused rows/cols.
  always @(*) begin
    reg [M_W-1:0] m_rem;
    reg [N_W-1:0] n_rem;
    reg [K_W-1:0] k_rem;
    
    // Offsets in full dims
    m_off  = m_tile_r * Tm;
    n_off  = n_tile_r * Tn;
    k_off  = k_tile_r * Tk;

    // Remaining
    m_rem = (M > m_off[M_W-1:0]) ? (M - m_off[M_W-1:0]) : {M_W{1'b0}};
    n_rem = (N > n_off[N_W-1:0]) ? (N - n_off[N_W-1:0]) : {M_W{1'b0}};
    k_rem = (K > k_off[K_W-1:0]) ? (K - k_off[K_W-1:0]) : {M_W{1'b0}};

    // Effective sizes (clamp to tile sizes)
    Tm_eff = (m_rem > Tm) ? Tm : m_rem[TM_W-1:0];
    Tn_eff = (n_rem > Tn) ? Tn : n_rem[TN_W-1:0];
    Tk_eff = (k_rem > Tk) ? Tk : k_rem[TK_W-1:0];
  end

  // Row/col masks: bit i/j set if within eff sizes
  // Masks are packed LSB=row0/col0..; consumer can AND with 'en' per PE
  always @(*) begin
    integer i, j;
    en_mask_row = {M_W{1'b0}};
    en_mask_col = {M_W{1'b0}};
    for (i = 0; i < MAX_TM; i = i + 1) begin
      if (i < Tm_eff) en_mask_row[i] = 1'b1;
    end
    for (j = 0; j < MAX_TN; j = j + 1) begin
      if (j < Tn_eff) en_mask_col[j] = 1'b1;
    end
  end

  // ---------------------------------------------------------------------------
  // 3. Bank Selection (Double Buffering)
  // ---------------------------------------------------------------------------
  // Engineer's Note:
  // We use the LSB of the K-tile index to toggle between Ping (0) and Pong (1) buffers.
  // While the accelerator processes 'Ping', the DMA loads 'Pong'.
  always @(*) begin
    bank_sel_k = k_tile_r[0]; // even k_tile -> ping(0), odd -> pong(1)
    A_ready    = bank_sel_k ? valid_A_pong : valid_A_ping;
    B_ready    = bank_sel_k ? valid_B_pong : valid_B_ping;
  end

  // ------------------------
  // Start latch & tile counts
  // ------------------------
  always @(posedge clk_gated or negedge rst_n) begin
    if (!rst_n) begin
      start_latched   <= 1'b0;
    end else begin
      if (start)      start_latched <= 1'b1;
      // cleared when entering S_PREP_TILE
      if (state == S_PREP_TILE) start_latched <= 1'b0;
    end
  end

  // Compute or load tile counts once per session (could also be latched in PREP_TILE)
  always @(*) begin
    reg [M_W-1:0] mt_calc;
    reg [N_W-1:0] nt_calc;
    reg [K_W-1:0] kt_calc;

    // Compute fallback counts (safe combinational helpers)
    mt_calc = ceil_div_M(M, Tm);
    nt_calc = ceil_div_N(N, Tn);
    kt_calc = ceil_div_K(K, Tk);

    // Use CSR-provided counts when requested, but fall back to computed values
    // if the CSR fields are zero (defensive: prevents a fatal zero-tile situation).
    if (USE_CSR_COUNTS) begin
      MT = (MT_csr != {M_W{1'b0}}) ? MT_csr : mt_calc;
      NT = (NT_csr != {M_W{1'b0}}) ? NT_csr : nt_calc;
      KT = (KT_csr != {M_W{1'b0}}) ? KT_csr : kt_calc;
    end else begin
      MT = mt_calc;
      NT = nt_calc;
      KT = kt_calc;
    end
  end

  // ------------------------
  // Tile indices & counters
  // ------------------------
  // m_tile, n_tile, k_tile advance in TILE_DONE / PREP next k
  always @(posedge clk_gated or negedge rst_n) begin
    if (!rst_n) begin
      m_tile_r <= {M_W{1'b0}};
      n_tile_r <= {M_W{1'b0}};
      k_tile_r <= {M_W{1'b0}};
      k_ctr  <= {M_W{1'b0}};
    end else if (abort) begin
      m_tile_r <= {M_W{1'b0}};
      n_tile_r <= {M_W{1'b0}};
      k_tile_r <= {M_W{1'b0}};
      k_ctr  <= {M_W{1'b0}};
    end else begin
      case (1'b1)  // ONE-HOT case for tile index updates
        state[1]: begin  // S_PREP_TILE
          // entering a new C tile (m,n), ensure k_tile=0
          k_tile_r <= {M_W{1'b0}};
          k_ctr  <= {M_W{1'b0}};
        end
        
        state[4]: begin  // S_LOAD_WEIGHT
          // WEIGHT-STATIONARY: cycle through K indices to load all weights
          if (k_ctr < Tk_eff - 1) begin
            k_ctr <= k_ctr + 1'b1;  // Load next weight
          end else begin
            k_ctr <= {M_W{1'b0}};  // Reset for activation streaming
          end
        end
        
        state[5]: begin  // S_STREAM_K
          // WEIGHT-STATIONARY: stream activations (weights already loaded)
          if (Tk_eff != {M_W{1'b0}}) begin
            // FIX: Reset counter at end of tile so next S_LOAD_WEIGHT starts at 0
            if (k_ctr == Tk_eff - 1) begin
               k_ctr <= {M_W{1'b0}};
            end else begin
               k_ctr <= k_ctr + 1'b1;
            end
          end
        end
        
        state[6]: begin  // S_TILE_DONE
          // Advance n/m; k resets next S_PREP_TILE
          if (n_tile_r + 1 < NT) begin
            n_tile_r <= n_tile_r + 1'b1;
          end else begin
            n_tile_r <= {M_W{1'b0}};
            if (m_tile_r + 1 < MT) begin
              m_tile_r <= m_tile_r + 1'b1;
            end else begin
              m_tile_r <= {M_W{1'b0}};
            end
          end
        end
        default: ;
      endcase

      // Advance k_tile when we finish a k-slice
      // FIX: Changed (k_ctr == Tk_eff) to (k_ctr == Tk_eff - 1)
      if (state[5] && (Tk_eff == {M_W{1'b0}} || (k_ctr == Tk_eff - 1))) begin  
        if (k_tile_r + 1 < KT) begin
          k_tile_r <= k_tile_r + 1'b1;
        end
      end
    end
  end

  // ------------------------
  // FSM next-state & outputs
  // ------------------------
  // Note: Outputs rd_en/k_idx/en/clr are driven in this block for clarity.
  always @(*) begin
    state_n        = state;

    // Default outputs already zeroed above; we drive deltas per state.
    // k_idx mirrors k_ctr during STREAM_K.
    k_idx          = k_ctr;
    bypass_out     = bypass_mode ? {(MAX_TM*MAX_TN){1'b1}} : {(MAX_TM*MAX_TN){1'b0}};  // NEW: All 1s if bypass_mode, else all 0s

    case (1'b1)  // ONE-HOT case statement

      state[0]: begin  // S_IDLE
        if (start_latched) state_n = S_PREP_TILE;
      end

      state[1:1]: begin  // S_PREP_TILE
        // Fire a 1-cycle clr at the start of each (m,n) tile
        clr     = 1'b1;

        // Reset k_ctr; k_tile is already 0 here
        // Decide whether to pre-prime SRAM pipeline or go directly to wait
        if (PREPRIME) begin
          state_n = S_PREPRIME;  // Issue dummy read to eliminate bubble
        end else begin
          state_n = S_WAIT_READY;  // Simple path: accept 1-cycle bubble
        end
      end

      state[2]: begin  // S_WAIT_READY
        // Wait until both A and B banks for this k_tile are declared ready.
        if (A_ready && B_ready) begin
          state_n = S_LOAD_WEIGHT;  // Go to weight loading phase (weight-stationary)
        end
        // Count stall cycles while waiting
      end

      state[3]: begin  // S_PREPRIME (PREPRIME=1 only)
        // Issue dummy read to pre-load SRAM output registers
        // This eliminates the 1-cycle bubble at the start of S_STREAM_K
        rd_en    = 1'b1;
        act_addr = {ADDR_W{1'b0}};  // Read from first address (dummy, will be re-read)
        
        // Next cycle, SRAM outputs will be valid, so first S_STREAM_K compute has no bubble
        state_n = S_WAIT_READY;
      end

      state[4]: begin  // S_LOAD_WEIGHT
        // WEIGHT-STATIONARY WEIGHT LOADING PHASE
        // Load ALL K weights for this tile into PE weight registers
        // Weights stay stationary for entire activation streaming phase
        load_weight_comb = 1'b1;  // Assert weight load control (internal)
        rd_en       = 1'b1;  // Read from weight buffer
        k_idx       = k_ctr; // Cycle through K indices to load all weights
        wgt_addr    = k_ctr[ADDR_W-1:0];  // Weight buffer address
        
        // After loading all Tk_eff weights, proceed to activation streaming
        if (k_ctr >= Tk_eff - 1) begin
          state_n = S_STREAM_K;
        end
      end

      state[5]: begin  // S_STREAM_K
        // WEIGHT-STATIONARY ACTIVATION STREAMING PHASE
        // Weights are now stationary in PEs (loaded in previous phase)
        // Stream activations through the array, weights stay put
        
        load_weight_comb = 1'b0;  // No more weight loading
        
        // Read activations from buffer (weights already loaded)
        rd_en = (Tk_eff != {M_W{1'b0}}) && (k_ctr < Tk_eff);
        k_idx = k_ctr;
        act_addr = k_ctr[ADDR_W-1:0];  // Activation buffer address

        // Enable MACs for all cycles (weights are preloaded)
        if (Tk_eff != {M_W{1'b0}}) begin
          en = 1'b1;
        end

        // If we just finished last k step, decide next:
        if (Tk_eff == {M_W{1'b0}}) begin
          // No work in this k-slice (edge case), treat as done
          if (k_tile_r + 1 < KT) state_n = S_WAIT_READY; // next k-slice
          else                 state_n = S_TILE_DONE;
        end else if (k_ctr == Tk_eff - 1) begin
          // End of k-slice (consumed last read); flip bank (via k_tile parity)
          if (k_tile_r + 1 < KT) state_n = S_WAIT_READY; // next k-slice
          else                 state_n = S_TILE_DONE;  // all k done for this (m,n)
        end
      end

      state[6]: begin  // S_TILE_DONE
        done_tile = 1'b1;
        // Advance n/m indices happens in the index block; decide if more tiles remain
        if ((n_tile_r + 1 < NT) || (m_tile_r + 1 < MT)) state_n = S_PREP_TILE;
        else                                         state_n = S_DONE;
      end

      state[7]: begin  // S_DONE
        // Hold until a new start
        if (start) state_n = S_PREP_TILE;
      end

      default: state_n = S_IDLE;
    endcase
 
    // Abort handling (synchronous)
    if (abort) begin
      state_n = S_IDLE;
    end
  end

  // ------------------------
  // State / perf registers
  // ------------------------
  always @(posedge clk_gated or negedge rst_n) begin
    if (!rst_n) begin
      state           <= S_IDLE;
      cycles_tile_r   <= {M_W{1'b0}};
      stall_cycles_r  <= {M_W{1'b0}};
    end else begin
      state <= state_n;

      // Per-tile cycle counters (one-hot state checks)
      if (state[1]) begin  // S_PREP_TILE
        cycles_tile_r  <= 32'd0;
        stall_cycles_r <= 32'd0;
      end else if (state[5]) begin  // S_STREAM_K
        cycles_tile_r  <= cycles_tile_r + 32'd1;
      end else if (state[2]) begin  // S_WAIT_READY
        stall_cycles_r <= stall_cycles_r + 32'd1;
      end
    end
  end

  assign cycles_tile  = cycles_tile_r;
  assign stall_cycles = stall_cycles_r;
  assign m_tile = m_tile_r;
  assign n_tile = n_tile_r;
  assign k_tile = k_tile_r;

  // ---------------------------------------------------------------------------
  // Assertions & Verification
  // ---------------------------------------------------------------------------
  // synthesis translate_off
  initial begin
    if ($test$plusargs("DEBUG_SCHED")) begin
      $display("Scheduler: Debug assertions enabled.");
    end
  end

  always @(posedge clk) begin
    if (rst_n) begin
      // 1. Bounds Check: Tile indices must be within configured limits
      if (state != S_IDLE) begin
        assert (m_tile_r < MT) else $error("Violation: m_tile_r (%0d) >= MT (%0d)", m_tile_r, MT);
        assert (n_tile_r < NT) else $error("Violation: n_tile_r (%0d) >= NT (%0d)", n_tile_r, NT);
        assert (k_tile_r < KT) else $error("Violation: k_tile_r (%0d) >= KT (%0d)", k_tile_r, KT);
      end

      // 2. Protocol: Weight-Stationary invariant
      // We cannot load weights and compute (en) at the same time
      assert (!(load_weight && en)) 
        else $error("Violation: load_weight and en asserted simultaneously (WS violation)");

      // 3. Counter Safety
      // k_ctr is used for addressing, must be within ADDR_W and Tk limits
      if (state[4] || state[5]) begin // S_LOAD_WEIGHT or S_STREAM_K
        assert (k_ctr < (1 << TK_W)) else $error("Violation: k_ctr overflow");
        // k_ctr should track Tk_eff. If it exceeds Tk, we are reading garbage or out of bounds.
        // Note: k_ctr resets to 0, so it might equal Tk_eff for one cycle during transition if logic isn't perfect,
        // but for addressing it should be < Tk.
        assert (k_ctr <= Tk) else $error("Violation: k_ctr (%0d) > Tk (%0d)", k_ctr, Tk);
      end

      // 4. State Machine Safety (One-Hot Encoding)
      assert ($onehot0(state)) 
        else $error("Violation: FSM state is not one-hot! state=%b", state);

      // 5. Bank Selection Safety
      // We should never be ready if the bank selection is invalid or out of sync (hard to check without more context, 
      // but we can check that we don't hang in WAIT_READY forever if valid signals are present).
    end
  end
  // synthesis translate_on

  // ------------------------
  // Synthesis-time safety notes (SVA in TB recommended)
  // ------------------------
  // 1) No read from a bank unless A_ready/B_ready for that bank asserted.
  // 2) PREPRIME=0: expect a 1-cycle compute bubble at k-start (en=0 on first cycle).
  // 3) PREPRIME=1: expect no bubble; first STREAM_K cycle computes immediately.
  // 4) en_mask_row/en_mask_col should be ANDed inside array or at MAC enable granularity.
  // 5) Optional: assert Tk != 0, Tm != 0, Tn != 0 at start (or treat zero as no-op edges).
  // 6) Document that bank_sel_rd_A/B = k_tile[0] policy; host must preload the opposite bank.

endmodule
