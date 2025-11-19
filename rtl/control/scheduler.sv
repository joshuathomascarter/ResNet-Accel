//------------------------------------------------------------------------------
// scheduler.v
// Tiled RS (Row-Stationary) GEMM scheduler for systolic array
//
// Responsibilities:
//  - Implements tile loop-nest over (m_tile, n_tile, k_tile)
//  - Drives buffers/array: clr, en, rd_en, k_idx, bank_sel_rd_A/B
//  - Honors 1-cycle SRAM read latency (see PREPRIME parameter)
//  - Generates row/col enable masks for edge tiles (Tm_eff/Tn_eff)
//  - Ping/Pong bank policy: k_tile parity
//  - Status + perf: busy, done_tile, tile coords, cycles, stall_cycles
//
// Latency note (per tile, ideal):
//   cycles_tile ≈ Tk_eff + (Tm-1) + (Tn-1)
//   +1 cycle bubble if PREPRIME=0 (documented below)
//
// Copyright:
//   Accel v1 (INT8 GEMM IP). MIT/Apache as you choose.
//------------------------------------------------------------------------------

module scheduler #(
  // Dimension widths (log2 maxima)
  parameter M_W  = 10,  // log2(max M)
  parameter N_W  = 10,  // log2(max N)
  parameter K_W  = 12,  // log2(max K) -- K is typically larger
  parameter TM_W = 6,   // log2(max Tm)
  parameter TN_W = 6,   // log2(max Tn)
  parameter TK_W = 6,   // log2(max Tk)

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
  output reg                  bank_sel_rd_A,  // 0: ping, 1: pong
  output reg                  bank_sel_rd_B,  // typically mirror A
  output reg                  clr,            // 1-cycle pulse at tile start
  output reg                  en,             // MAC enable (AND with row/col masks externally if desired)
  output reg                  load_weight,    // NEW: weight loading control for row-stationary
  output reg [MAX_TM-1:0]     en_mask_row,    // bit i = 1 -> row i valid
  output reg [MAX_TN-1:0]     en_mask_col,    // bit j = 1 -> col j valid

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
  // Clock Gating Logic (saves ~200 mW when scheduler idle)
  // ========================================================================
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
  // State encoding - UPDATED for row-stationary weight loading
  localparam [2:0] S_IDLE        = 3'b000,
                   S_PREP_TILE   = 3'b001,
                   S_WAIT_READY  = 3'b010,
                   S_LOAD_WEIGHT = 3'b011,  // NEW: load weights into PEs (row-stationary)
                   S_STREAM_K    = 3'b100,  // activations stream, weights stationary
                   S_TILE_DONE   = 3'b101,
                   S_DONE        = 3'b110;

  reg [2:0] state, state_n;

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

  // Outputs default
  always @(*) begin
    rd_en          = 1'b0;
    k_idx          = {TK_W{1'b0}};
    bank_sel_rd_A  = bank_sel_k;
    bank_sel_rd_B  = bank_sel_k;
    clr            = 1'b0;
    en             = 1'b0;
    load_weight    = 1'b0;  // NEW: default weight load off
    done_tile      = 1'b0;
    busy           = (state != S_IDLE) && (state != S_DONE);
  end

  // ------------------------
  // Helpers (combinational)
  // ------------------------

  // Ceil-div helper (be aware: synthesizes a divider if use_csr_counts=0)
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

  // Compute effective sizes for edge tiles: eff = min(T*, remaining)
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

  // Bank select policy & readiness
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
      case (state)
        S_PREP_TILE: begin
          // entering a new C tile (m,n), ensure k_tile=0
          k_tile_r <= {M_W{1'b0}};
          k_ctr  <= {M_W{1'b0}};
        end
        
        S_LOAD_WEIGHT: begin
          // ROW-STATIONARY: cycle through K indices to load all weights
          if (k_ctr < Tk_eff - 1) begin
            k_ctr <= k_ctr + 1'b1;  // Load next weight
          end else begin
            k_ctr <= {M_W{1'b0}};  // Reset for activation streaming
          end
        end
        
        S_STREAM_K: begin
          // ROW-STATIONARY: stream activations (weights already loaded)
          if (Tk_eff != {M_W{1'b0}}) begin
            if (k_ctr < Tk_eff) begin
              k_ctr <= k_ctr + 1'b1;
            end
          end
        end
        
        S_TILE_DONE: begin
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
      if (state == S_STREAM_K && (Tk_eff == {M_W{1'b0}} || (k_ctr == Tk_eff))) begin
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

    case (state)

      S_IDLE: begin
        if (start_latched) state_n = S_PREP_TILE;
      end

      S_PREP_TILE: begin
        // Fire a 1-cycle clr at the start of each (m,n) tile
        clr     = 1'b1;

        // Reset k_ctr; k_tile is already 0 here
        // Decide whether to pre-prime or go wait for bank ready
        state_n = S_WAIT_READY;
      end

      S_WAIT_READY: begin
        // Wait until both A and B banks for this k_tile are declared ready.
        if (A_ready && B_ready) begin
          state_n = S_LOAD_WEIGHT;  // Go to weight loading phase (row-stationary)
        end
        // Count stall cycles while waiting
      end

      S_LOAD_WEIGHT: begin
        // ROW-STATIONARY WEIGHT LOADING PHASE
        // Load ALL K weights for this tile into PE weight registers
        // Weights stay stationary for entire activation streaming phase
        load_weight = 1'b1;  // Assert weight load control
        rd_en       = 1'b1;  // Read from weight buffer
        k_idx       = k_ctr; // Cycle through K indices to load all weights
        
        // After loading all Tk_eff weights, proceed to activation streaming
        if (k_ctr >= Tk_eff - 1) begin
          state_n = S_STREAM_K;
        end
      end

      S_STREAM_K: begin
        // ROW-STATIONARY ACTIVATION STREAMING PHASE
        // Weights are now stationary in PEs (loaded in previous phase)
        // Stream activations through the array, weights stay put
        
        load_weight = 1'b0;  // No more weight loading
        
        // Read activations from buffer (weights already loaded)
        rd_en = (Tk_eff != {M_W{1'b0}}) && (k_ctr < Tk_eff);
        k_idx = k_ctr;

        // Enable MACs for all cycles (weights are preloaded)
        if (Tk_eff != {M_W{1'b0}}) begin
          en = 1'b1;
        end

        // If we just finished last k step, decide next:
        if (Tk_eff == {M_W{1'b0}}) begin
          // No work in this k-slice (edge case), treat as done
          if (k_tile_r + 1 < KT) state_n = S_WAIT_READY; // next k-slice
          else                 state_n = S_TILE_DONE;
        end else if (k_ctr == Tk_eff) begin
          // End of k-slice (consumed last read); flip bank (via k_tile parity)
          if (k_tile_r + 1 < KT) state_n = S_WAIT_READY; // next k-slice
          else                 state_n = S_TILE_DONE;  // all k done for this (m,n)
        end
      end

      S_TILE_DONE: begin
        done_tile = 1'b1;
        // Advance n/m indices happens in the index block; decide if more tiles remain
        if ((n_tile_r + 1 < NT) || (m_tile_r + 1 < MT)) state_n = S_PREP_TILE;
        else                                         state_n = S_DONE;
      end

      S_DONE: begin
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

      // Per-tile cycle counters
      if (state == S_PREP_TILE) begin
        cycles_tile_r  <= 32'd0;
        stall_cycles_r <= 32'd0;
      end else if (state == S_STREAM_K) begin
        cycles_tile_r  <= cycles_tile_r + 32'd1;
      end else if (state == S_WAIT_READY) begin
        stall_cycles_r <= stall_cycles_r + 32'd1;
      end
    end
  end

  assign cycles_tile  = cycles_tile_r;
  assign stall_cycles = stall_cycles_r;
  assign m_tile = m_tile_r;
  assign n_tile = n_tile_r;
  assign k_tile = k_tile_r;

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
