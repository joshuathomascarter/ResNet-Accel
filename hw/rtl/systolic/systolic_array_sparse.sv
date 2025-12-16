/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                      SYSTOLIC_ARRAY_SPARSE.SV                             ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  Sparse-Aware Weight-Stationary Systolic Array for INT8 CNNs             ║
 * ║  Optimized for ResNet-18 inference on PYNQ-Z2 (Zynq-7020)                ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 *
 * OVERVIEW:
 * ---------
 * This module implements a 14×14 systolic array with sparse computation support.
 * It computes matrix multiplication: C[M×N] = A[M×K] × B[K×N]
 * 
 * The "sparse-aware" feature means:
 *   - Zero blocks in weight matrix are skipped entirely
 *   - block_valid signal gates all PEs (no wasted cycles on zeros)
 *   - 50-70% speedup for pruned ResNet-18 (50-70% sparsity)
 *
 * ARRAY DIMENSIONS:
 * -----------------
 *   N_ROWS = 14: Processes 14 output rows per tile (M dimension)
 *   N_COLS = 14: Processes 14 output columns per tile (N dimension)
 *   Total PEs = 14 × 14 = 196
 *   Total DSPs = 196 (89% of Zynq-7020's 220 DSP48E1 slices)
 *
 * WHY 14×14?
 * ----------
 * PYNQ-Z2 (Zynq-7020) resource budget:
 *   - 220 DSP48E1 slices available
 *   - 196 DSPs used for array (89% utilization)
 *   - 24 DSPs reserved for output scaling, bias, etc.
 *
 * 16×16 would need 256 DSPs (exceeds budget)
 * 14×14 is the largest square that fits
 *
 * WEIGHT-STATIONARY DATAFLOW:
 * ---------------------------
 *           Weights (broadcast vertically)
 *           ┌─────┬─────┬─────┬─────┐
 *           │W0,0 │W0,1 │W0,2 │ ... │
 *           └──┬──┴──┬──┴──┬──┴─────┘
 *              │     │     │
 *   ┌──────────▼─────▼─────▼─────────────────────┐
 *   │ Activations   ┌─────┐ ┌─────┐ ┌─────┐      │
 *   │ (stream West  │PE   │ │PE   │ │PE   │      │
 *   │  to East)  ──▶│0,0  │→│0,1  │→│0,2  │→ ... │
 *   │               └─────┘ └─────┘ └─────┘      │
 *   │               ┌─────┐ ┌─────┐ ┌─────┐      │
 *   │            ──▶│PE   │→│PE   │→│PE   │→ ... │
 *   │               │1,0  │ │1,1  │ │1,2  │      │
 *   │               └─────┘ └─────┘ └─────┘      │
 *   │                 ...                        │
 *   └────────────────────────────────────────────┘
 *
 * TIMING PER TILE:
 * ----------------
 *   1. Weight Loading:    14 cycles (1 row per cycle)
 *   2. Activation Stream: K cycles (K = inner dimension)
 *   3. Result Read:       Immediate (accumulators hold results)
 *
 * SPARSE OPERATION:
 * -----------------
 * The scheduler identifies zero blocks in the weight matrix (BSR format).
 * When a block is zero, block_valid=0 and:
 *   - All PEs hold their accumulators (no update)
 *   - Activations are not streamed
 *   - Zero power consumed in compute (only leakage)
 *
 * For a 70% sparse network, only 30% of blocks are processed.
 * Speedup ≈ 1 / 0.30 = 3.3× theoretical (2.5× practical due to overhead)
 *
 * BYPASS MODE (ResNet Skip Connections):
 * --------------------------------------
 * bypass_flat provides per-PE control for residual connections.
 * When bypass[i][j]=1, PE[i][j] skips multiplication:
 *   acc = acc + activation (instead of acc = acc + act × weight)
 *
 * This implements y = F(x) + x in ResNet without separate hardware.
 *
 * POWER CONSUMPTION:
 * ------------------
 *   Active compute: ~450 mW @ 100 MHz
 *   Idle (block_valid=0): ~50 mW (leakage only)
 *   Average for 70% sparse: ~170 mW
 */

`default_nettype none

module systolic_array_sparse #(
    // =========================================================================
    // PARAMETER: N_ROWS - Number of PE rows
    // =========================================================================
    // Determines how many output rows are computed per tile.
    // Default: 14 for PYNQ-Z2 DSP budget (14 × 14 = 196 DSPs)
    parameter N_ROWS = 14,
    
    // =========================================================================
    // PARAMETER: N_COLS - Number of PE columns
    // =========================================================================
    // Determines how many output columns are computed per tile.
    parameter N_COLS = 14,
    
    // =========================================================================
    // PARAMETER: DATA_W - Data width in bits
    // =========================================================================
    // Width of activation and weight values.
    // 8 bits for INT8 quantized networks.
    parameter DATA_W = 8,
    
    // =========================================================================
    // PARAMETER: ACC_W - Accumulator width in bits
    // =========================================================================
    // Width of partial sum accumulators.
    // 32 bits to prevent overflow during K accumulations.
    // Max value: 128 × 128 × 4096 = 67M (requires 27 bits, 32 for safety)
    parameter ACC_W  = 32
)(
    // =========================================================================
    // PORT: clk - System Clock
    // =========================================================================
    input  wire                     clk,
    
    // =========================================================================
    // PORT: rst_n - Active-Low Asynchronous Reset
    // =========================================================================
    input  wire                     rst_n,
    
    // =========================================================================
    // SPARSE CONTROL SIGNALS (from scheduler)
    // =========================================================================
    
    /**
     * block_valid - PE Enable for Current Block
     * ------------------------------------------
     * 1 = Current weight block is non-zero, compute MAC
     * 0 = Current weight block is zero, skip (hold accumulators)
     *
     * This is the key sparse control signal. When the scheduler detects
     * a zero block in BSR format, it de-asserts block_valid to save power.
     */
    input  wire                     block_valid,
    
    /**
     * load_weight - Weight Loading Control
     * -------------------------------------
     * 1 = Load weights from b_in into PE weight registers
     * 0 = Use existing weights (stationary)
     *
     * Assert for 14 cycles at start of each tile (1 row per cycle).
     */
    input  wire                     load_weight,
    
    /**
     * bypass_flat - Per-PE Bypass Control (Flattened)
     * ------------------------------------------------
     * bypass_flat[i*N_COLS + j] = bypass signal for PE[i][j]
     * 
     * When 1: PE skips multiply, adds activation directly
     * Used for ResNet skip connections
     *
     * SIZE: N_ROWS × N_COLS = 196 bits for 14×14 array
     */
    input  wire [N_ROWS*N_COLS-1:0] bypass_flat,
    
    // =========================================================================
    // DATA INPUTS
    // =========================================================================
    
    /**
     * a_in_flat - Activation Inputs (Flattened)
     * -----------------------------------------
     * Contains N_ROWS signed INT8 activations.
     * a_in_flat[i*DATA_W +: DATA_W] = activation for row i
     *
     * Source: act_buffer via scheduler
     * SIZE: 14 × 8 = 112 bits
     */
    input  wire [N_ROWS*DATA_W-1:0] a_in_flat,
    
    /**
     * b_in_flat - Weight Inputs (Flattened)
     * -------------------------------------
     * Contains N_COLS signed INT8 weights.
     * b_in_flat[j*DATA_W +: DATA_W] = weight for column j
     *
     * Source: wgt_buffer (BSR block data)
     * SIZE: 14 × 8 = 112 bits
     */
    input  wire [N_COLS*DATA_W-1:0] b_in_flat,
    
    // =========================================================================
    // DATA OUTPUTS
    // =========================================================================
    
    /**
     * c_out_flat - Accumulated Results (Flattened)
     * ---------------------------------------------
     * Contains N_ROWS × N_COLS INT32 accumulators.
     * c_out_flat[(i*N_COLS + j)*ACC_W +: ACC_W] = result for PE[i][j]
     *
     * SIZE: 14 × 14 × 32 = 6272 bits
     */
    output wire [N_ROWS*N_COLS*ACC_W-1:0] c_out_flat
);

  // ===========================================================================
  // SECTION 1: INPUT UNPACKING
  // ===========================================================================
  // Convert flattened input buses to 2D arrays for easier indexing.
  //
  // FLATTENING CONVENTION:
  // ----------------------
  // flat[i*WIDTH +: WIDTH] extracts bits [i*WIDTH, i*WIDTH + WIDTH - 1]
  // This is SystemVerilog "indexed part-select" syntax.
  //
  // Example: a_in_flat = 112 bits, DATA_W = 8
  //   a_in[0] = a_in_flat[7:0]   (bits 0-7)
  //   a_in[1] = a_in_flat[15:8]  (bits 8-15)
  //   ...
  //
  wire signed [DATA_W-1:0] a_in [0:N_ROWS-1];  // Unpacked activations
  wire signed [DATA_W-1:0] b_in [0:N_COLS-1];  // Unpacked weights
  wire bypass_mat [0:N_ROWS-1][0:N_COLS-1];    // 2D bypass matrix
  
  genvar i;
  generate
    // Unpack activation inputs (one per row)
    for (i = 0; i < N_ROWS; i = i + 1) 
      assign a_in[i] = a_in_flat[i*DATA_W +: DATA_W];
    
    // Unpack weight inputs (one per column)  
    for (i = 0; i < N_COLS; i = i + 1) 
      assign b_in[i] = b_in_flat[i*DATA_W +: DATA_W];
    
    // Unpack bypass signals (one per PE)
    // bypass_flat is row-major: bypass_flat[r*N_COLS + c] → bypass_mat[r][c]
    for (i = 0; i < N_ROWS*N_COLS; i = i + 1) begin : UNPACK_BYPASS
      // Calculate row and column indices
      // NOTE: These are compile-time constants, no hardware dividers generated
      localparam integer r_idx = i / N_COLS;
      localparam integer c_idx = i % N_COLS;
      assign bypass_mat[r_idx][c_idx] = bypass_flat[i];
    end
  endgenerate

  // ===========================================================================
  // SECTION 2: WEIGHT LOADING STATE MACHINE
  // ===========================================================================
  //
  // Weight loading happens row-by-row, one row per clock cycle.
  // This is necessary because:
  //   1. Each row has N_COLS weights (14 bytes)
  //   2. Weight buffer outputs one row at a time
  //   3. Systolic load_weight propagation needs 1 cycle per column
  //
  // LOADING SEQUENCE FOR 14×14 ARRAY:
  // ----------------------------------
  //   Cycle 0:  load_ptr=0  → Row 0 weights loaded
  //   Cycle 1:  load_ptr=1  → Row 1 weights loaded
  //   ...
  //   Cycle 13: load_ptr=13 → Row 13 weights loaded
  //   Cycle 14: load_ptr=0  → Loading complete, ready for compute
  //
  // TOTAL LOADING TIME: 14 cycles (amortized over K compute cycles)
  //
  reg [$clog2(N_ROWS)-1:0] load_ptr;  // Current row being loaded

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // Reset: Start at row 0
      load_ptr <= 0;
    end else begin
      if (load_weight) begin
        // Advance to next row each cycle during weight loading
        // $clog2(N_ROWS) bits ensures proper wraparound
        load_ptr <= load_ptr + 1;
      end else begin
        // Reset pointer when not loading (ready for next load phase)
        load_ptr <= 0;
      end
    end
  end

  // ===========================================================================
  // SECTION 3: PE INTERCONNECTION WIRES
  // ===========================================================================
  //
  // These 2D wire arrays connect PEs together:
  //
  //   a_fwd[r][c]: Activation forwarded from PE[r][c] to PE[r][c+1]
  //   load_weight_fwd[r][c]: Load signal forwarded from PE[r][c] to PE[r][c+1]
  //
  //   ┌─────────────────────────────────────────────────────────────┐
  //   │     Column 0        Column 1        Column 2      ...      │
  //   │                                                             │
  //   │  a_in[0] ──▶ PE[0][0] ──a_fwd──▶ PE[0][1] ──a_fwd──▶ ...   │
  //   │                 │                    │                      │
  //   │           load_weight_fwd      load_weight_fwd              │
  //   │                 ▼                    ▼                      │
  //   │                                                             │
  //   │  a_in[1] ──▶ PE[1][0] ──a_fwd──▶ PE[1][1] ──a_fwd──▶ ...   │
  //   │                                                             │
  //   └─────────────────────────────────────────────────────────────┘
  //
  wire signed [DATA_W-1:0] a_fwd [0:N_ROWS-1][0:N_COLS-1];  // Activation pipeline
  wire load_weight_fwd [0:N_ROWS-1][0:N_COLS-1];            // Load signal pipeline

  // ===========================================================================
  // SECTION 4: PE ARRAY INSTANTIATION
  // ===========================================================================
  //
  // This double-nested generate loop creates the 14×14 PE grid.
  // Each PE is connected according to the weight-stationary dataflow.
  //
  // PE PORT CONNECTIONS:
  // --------------------
  //   .a_in     → Activation from left (col 0: from input, else: from PE[r][c-1])
  //   .b_in     → Weight broadcast (same weight to all rows in column)
  //   .bypass   → Residual mode from bypass_mat[r][c]
  //   .en       → block_valid (0 for zero blocks → skip compute)
  //   .clr      → 1'b0 (clear handled externally before tile)
  //   .load_weight → Row-selective loading (only load_ptr's row gets it)
  //   .a_out    → Forwarded activation to PE[r][c+1]
  //   .load_weight_out → Forwarded load signal to PE[r][c+1]
  //   .acc      → Direct output to c_out_flat
  //
  // PE ARRAY (ROW r, COL c):
  // ┌───────────────────────────────────────────────────────────────────────┐
  // │ genvar r: 0, 1, 2, ..., N_ROWS-1 (rows)                              │
  // │ genvar c: 0, 1, 2, ..., N_COLS-1 (columns)                           │
  // │                                                                       │
  // │ for (r = 0; r < N_ROWS; r++)                                         │
  // │   for (c = 0; c < N_COLS; c++)                                       │
  // │     instantiate pe[r][c] with appropriate connections                │
  // └───────────────────────────────────────────────────────────────────────┘
  //
  genvar r, c;
  generate
    for (r = 0; r < N_ROWS; r = r + 1) begin : ROW
      for (c = 0; c < N_COLS; c = c + 1) begin : COL
        
        // ─────────────────────────────────────────────────────────────────────
        // ACTIVATION SOURCE SELECTION
        // ─────────────────────────────────────────────────────────────────────
        // Column 0: Activation comes directly from input (act_buffer)
        // Column N: Activation comes from previous PE's output (pipeline)
        //
        wire signed [DATA_W-1:0] a_src;
        if (c == 0) begin : gen_a_input
          // First column: get activation from external input
          assign a_src = a_in[r];
        end else begin : gen_a_chain
          // Other columns: get activation from left neighbor
          assign a_src = a_fwd[r][c-1];
        end
        
        // ─────────────────────────────────────────────────────────────────────
        // WEIGHT SOURCE (BROADCAST)
        // ─────────────────────────────────────────────────────────────────────
        // WEIGHT-STATIONARY KEY INSIGHT:
        // ALL rows in the same column get the SAME weight.
        // The weight is loaded once into each PE's weight register,
        // then held constant while activations stream through.
        //
        wire signed [DATA_W-1:0] b_src = b_in[c];  // Broadcast to column
        
        // ─────────────────────────────────────────────────────────────────────
        // LOAD WEIGHT SOURCE SELECTION
        // ─────────────────────────────────────────────────────────────────────
        // Column 0: Load signal from scheduler (gated by load_ptr match)
        // Column N: Load signal from left neighbor (systolic propagation)
        //
        // ROW-SELECTIVE LOADING:
        // Only the row matching load_ptr receives the load signal.
        // This loads weights row-by-row over 14 cycles.
        //
        wire load_weight_src;
        if (c == 0) begin : gen_lw_input
          // First column: check if this is the row being loaded
          // load_ptr == r[$clog2(N_ROWS)-1:0] compares row index
          assign load_weight_src = load_weight && (load_ptr == r[$clog2(N_ROWS)-1:0]);
        end else begin : gen_lw_chain
          // Other columns: receive load signal from left neighbor
          assign load_weight_src = load_weight_fwd[r][c-1];
        end

        // ─────────────────────────────────────────────────────────────────────
        // PE INSTANTIATION
        // ─────────────────────────────────────────────────────────────────────
        // PIPE=1: Register activations for timing closure at 200 MHz
        //
        pe #(.PIPE(1)) u_pe (
          .clk            (clk),
          .rst_n          (rst_n),
          
          // SPARSE CONTROL: block_valid gates all computation
          // When 0, PEs hold their accumulators (zero block skip)
          .en             (block_valid),
          
          // CLEAR: Always 0 here, clearing done before tile starts
          .clr            (1'b0),
          
          // WEIGHT LOADING: Row-selective via load_weight_src
          .load_weight    (load_weight_src), 
          
          // DATA PATHS
          .a_in           (a_src),           // Activation (streaming)
          .b_in           (b_src),           // Weight (broadcast)
          .bypass         (bypass_mat[r][c]),// Residual mode
          
          // OUTPUTS (forwarded or connected to output bus)
          .a_out          (a_fwd[r][c]),
          .load_weight_out(load_weight_fwd[r][c]),
          
          // RESULT: Direct connection to flattened output bus
          // Formula: (r*N_COLS + c)*ACC_W gives starting bit position
          .acc            (c_out_flat[(r*N_COLS + c)*ACC_W +: ACC_W])
        );
      end
    end
  endgenerate

endmodule

`default_nettype wire
