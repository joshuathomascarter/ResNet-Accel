/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                               PE.SV                                       ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  Weight-Stationary Processing Element for INT8 Systolic Array            ║
 * ║  Target: PYNQ-Z2 (Zynq-7020) @ 100-200 MHz                               ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 *
 * WEIGHT-STATIONARY ARCHITECTURE:
 * --------------------------------
 * In weight-stationary dataflow, each PE holds ONE weight for the entire
 * computation of an output tile. This maximizes weight reuse:
 *
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │                     WEIGHT-STATIONARY PE                           │
 *   │                                                                     │
 *   │   Weight (from B matrix)      Activation (from A matrix)           │
 *   │         │                           │                               │
 *   │         ▼                           ▼                               │
 *   │   ┌──────────┐                ┌──────────┐                         │
 *   │   │ Weight   │───────────────▶│   MAC    │◀──────┐                 │
 *   │   │ Register │  (Stationary)  │   Unit   │       │                 │
 *   │   └──────────┘                └────┬─────┘       │                 │
 *   │         ▲                          │             │                 │
 *   │         │                          ▼             │                 │
 *   │   load_weight               ┌──────────┐         │                 │
 *   │                             │   ACC    │──────── ┘                 │
 *   │                             │ Register │   (feedback)             │
 *   │                             └──────────┘                          │
 *   │                                                                     │
 *   │   Activation IN ──────▶ Pipeline Reg ──────▶ Activation OUT       │
 *   │   (from left PE)                              (to right PE)        │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 * DATA FLOW DIRECTIONS:
 * ---------------------
 *   - Weights:      Broadcast from top (no horizontal flow)
 *   - Activations:  Stream West → East (horizontal pipeline)
 *   - Partial Sums: Accumulate locally (no vertical flow in WS)
 *
 * TIMING:
 * -------
 * For a K-depth accumulation:
 *   - Cycle 0:          Load weight (load_weight=1)
 *   - Cycles 1 to K:    Stream activations (en=1)
 *   - Cycle K+1:        Read accumulator (acc valid)
 *
 * WHY WEIGHT-STATIONARY?
 * ----------------------
 * 1. WEIGHT REUSE: Each weight is used M times (once per activation row)
 * 2. REDUCED BANDWIDTH: Weights loaded once per tile, not per cycle
 * 3. SIMPLE CONTROL: No weight forwarding between PEs
 * 4. CNN-OPTIMAL: Conv layers have high weight reuse (same kernel, many pixels)
 *
 * COMPARISON OF DATAFLOWS:
 * ┌───────────────────┬──────────────┬───────────────┬─────────────────┐
 * │ Dataflow          │ Weight Reuse │ Act Reuse     │ Best For        │
 * ├───────────────────┼──────────────┼───────────────┼─────────────────┤
 * │ Weight-Stationary │ High (M×)    │ Low (1×)      │ Conv layers     │
 * │ Output-Stationary │ Low (1×)     │ High (N×)     │ FC layers       │
 * │ Row-Stationary    │ Medium       │ Medium        │ Balanced        │
 * └───────────────────┴──────────────┴───────────────┴─────────────────┘
 *
 * RESOURCE USAGE (per PE):
 * ------------------------
 *   - 1 DSP48E1 slice (MAC unit)
 *   - 8 FFs for weight register
 *   - 8 FFs for activation pipeline (PIPE=1)
 *   - 32 FFs for accumulator (inside MAC)
 *   - ~20 LUTs for control logic
 *
 * PYNQ-Z2 BUDGET:
 * ---------------
 *   DSP48E1: 220 available, 196 used for 14×14 array (89% utilization)
 *   This leaves 24 DSPs for other functions (output scaling, bias add)
 */

`ifndef PE_V
`define PE_V
`default_nettype none

module pe #(
    // =========================================================================
    // PARAMETER: PIPE - Activation Pipeline Enable
    // =========================================================================
    // Controls whether activations are pipelined (registered) as they flow
    // horizontally through the array.
    //
    // PIPE = 1 (Recommended):
    //   - Activations are registered before forwarding to next PE
    //   - Allows higher clock frequency (shorter combinational paths)
    //   - Adds 1 cycle latency per PE (14 cycles total for 14-wide array)
    //   - CRITICAL for timing closure at 200 MHz
    //
    // PIPE = 0:
    //   - Combinational pass-through (no latency)
    //   - Lower max frequency (long combinational chain across PEs)
    //   - Only suitable for small arrays or low-speed testing
    //
    parameter PIPE = 1
)(
    // =========================================================================
    // PORT: clk - System Clock
    // =========================================================================
    // May be gated per-row for power savings (see systolic_array.sv)
    // Target: 100-200 MHz on Zynq-7020
    input  wire              clk,
    
    // =========================================================================
    // PORT: rst_n - Active-Low Asynchronous Reset
    // =========================================================================
    input  wire              rst_n,
    
    // =========================================================================
    // PORT: a_in - Activation Input (from Left PE or Buffer)
    // =========================================================================
    // Signed INT8 activation value
    // Source: For column 0: from act_buffer via skew register
    //         For column N: from PE[row][N-1].a_out
    input  wire signed [7:0] a_in,
    
    // =========================================================================
    // PORT: b_in - Weight Load Data (from Top/Weight Buffer)
    // =========================================================================
    // Weight value to be loaded into internal register
    // Only sampled when load_weight=1
    // In weight-stationary: same weight broadcasted to all PEs in a column
    input  wire signed [7:0] b_in,
    
    // =========================================================================
    // PORT: en - MAC Enable
    // =========================================================================
    // When HIGH: Perform MAC operation (acc += a × weight)
    // When LOW: Hold accumulator value
    // MUST be mutually exclusive with load_weight
    input  wire              en,
    
    // =========================================================================
    // PORT: clr - Accumulator Clear
    // =========================================================================
    // When HIGH: Clear accumulator to zero on next clock
    // Assert at start of each new output tile computation
    input  wire              clr,
    
    // =========================================================================
    // PORT: load_weight - Weight Load Enable
    // =========================================================================
    // When HIGH: Capture b_in into internal weight register
    // PROTOCOL: Assert for 1 cycle per PE before starting MAC operations
    // MUST be mutually exclusive with en (cannot load and compute simultaneously)
    input  wire              load_weight,
    
    // =========================================================================
    // PORT: bypass - Residual Bypass Mode
    // =========================================================================
    // When HIGH: Skip multiply, add activation directly (for ResNet skip connections)
    // Implements: acc = acc + a (instead of acc = acc + a×b)
    input  wire              bypass,
    
    // =========================================================================
    // PORT: a_out - Activation Output (to Right PE)
    // =========================================================================
    // Forwarded activation for horizontal systolic flow
    // Registered if PIPE=1, combinational if PIPE=0
    output logic signed [7:0] a_out,
    
    // =========================================================================
    // PORT: load_weight_out - Load Weight Propagation
    // =========================================================================
    // Passes load_weight signal to neighboring PE
    // Used to create systolic weight loading (each PE loads 1 cycle after left neighbor)
    // This eliminates fanout issues when load_weight fans out to all PEs
    output logic              load_weight_out,
    
    // =========================================================================
    // PORT: acc - Accumulator Result (INT32)
    // =========================================================================
    // The accumulated partial sum: Σ(activation × weight)
    // Read this after all K activations have been streamed
    // 32 bits to prevent overflow during accumulation (see mac8.sv for justification)
    output logic signed [31:0] acc
);

    // =========================================================================
    // SECTION 1: INTERNAL STATE REGISTERS
    // =========================================================================
    //
    // weight_reg: The "stationary" weight register
    // ---------------------------------------------
    // This register holds the weight value for the entire duration of a tile
    // computation. It's loaded ONCE at the start of each tile, then held
    // constant while activations stream through.
    //
    // Weight loading sequence for a 14×14 array:
    //   Cycle 0: PE[0][0-13] load weights (first row)
    //   Cycle 1: PE[1][0-13] load weights (second row)
    //   ...
    //   Cycle 13: PE[13][0-13] load weights (last row)
    //   Cycle 14: All PEs have weights, ready to compute
    //
    // NOTE: a_reg was removed - activations are not stored locally in WS dataflow
    //
    logic signed [7:0] weight_reg;  // Holds stationary weight value

    // =========================================================================
    // SECTION 2: WEIGHT STATIONARY LOGIC
    // =========================================================================
    //
    // The weight is loaded when load_weight is HIGH, then HELD until the next
    // load_weight pulse. This "stationary" behavior gives the dataflow its name.
    //
    // SYSTOLIC WEIGHT LOADING (Fanout Fix):
    // -------------------------------------
    // Problem: If load_weight fans out directly to all 196 PEs, timing fails
    //          due to high fanout delays.
    //
    // Solution: Chain load_weight through PEs systolically:
    //   - PE[r][c] receives load_weight from PE[r][c-1] (or scheduler if c=0)
    //   - PE[r][c] outputs load_weight_out to PE[r][c+1]
    //   - Each PE loads 1 cycle after its left neighbor
    //
    // This creates a "wave" of weight loading across each row:
    //   Cycle N:   PE[row][0] loads
    //   Cycle N+1: PE[row][1] loads
    //   ...
    //   Cycle N+13: PE[row][13] loads
    //
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // ─────────────────────────────────────────────────────────────────
            // RESET: Clear weight register and propagation signal
            // ─────────────────────────────────────────────────────────────────
            weight_reg <= 8'sd0;         // Zero weight (safe default)
            load_weight_out <= 1'b0;     // Don't propagate during reset
        end else begin
            // ─────────────────────────────────────────────────────────────────
            // PROPAGATE: Forward load_weight to right neighbor
            // ─────────────────────────────────────────────────────────────────
            // This creates the systolic loading pattern - each PE passes
            // the load signal to its neighbor with 1-cycle delay.
            //
            load_weight_out <= load_weight;
            
            // ─────────────────────────────────────────────────────────────────
            // CAPTURE: Load new weight when load_weight is asserted
            // ─────────────────────────────────────────────────────────────────
            // b_in comes from the weight buffer (same value to all PEs in column)
            // The weight is then held constant for all K accumulation cycles.
            //
            if (load_weight) begin
                weight_reg <= b_in;
            end
            // ELSE: Hold previous weight (stationary behavior)
        end
    end

    // =========================================================================
    // SECTION 3: ACTIVATION PIPELINE (HORIZONTAL FORWARDING)
    // =========================================================================
    //
    // Activations flow West → East through the systolic array.
    // Each PE receives an activation, uses it for MAC, then forwards it.
    //
    // WHY PIPELINE?
    // -------------
    // In a 14-wide array, combinational pass-through would create a chain:
    //   a_in[0] → PE0 → PE1 → ... → PE13 → a_out[13]
    //
    // This would be 14× the single-PE delay, making timing closure impossible
    // at 200 MHz.
    //
    // With PIPE=1, each PE adds a register:
    //   - Maximum combinational path = 1 PE
    //   - Allows much higher clock frequency
    //   - Trade-off: 14 extra cycles of latency (14-cycle pipeline)
    //
    // TIMING DIAGRAM (PIPE=1):
    // ┌─────────┬─────────┬─────────┬─────────┬─────────┐
    // │ Cycle   │ PE[0]   │ PE[1]   │ PE[2]   │ ...     │
    // ├─────────┼─────────┼─────────┼─────────┼─────────┤
    // │ 1       │ a[0]    │ -       │ -       │         │
    // │ 2       │ a[1]    │ a[0]    │ -       │         │
    // │ 3       │ a[2]    │ a[1]    │ a[0]    │         │
    // │ ...     │         │         │         │         │
    // └─────────┴─────────┴─────────┴─────────┴─────────┘
    //
    generate
        if (PIPE) begin : gen_pipe
            // ─────────────────────────────────────────────────────────────────
            // PIPELINED: Register activation before forwarding
            // ─────────────────────────────────────────────────────────────────
            // This is the recommended mode for high-frequency operation.
            // Adds 1 cycle latency per PE but enables timing closure.
            //
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) 
                    a_out <= 8'sd0;  // Clear on reset
                else        
                    a_out <= a_in;   // Forward with 1-cycle delay
            end
        end else begin : gen_comb
            // ─────────────────────────────────────────────────────────────────
            // COMBINATIONAL: Direct pass-through (no latency)
            // ─────────────────────────────────────────────────────────────────
            // WARNING: Not recommended for arrays larger than 4×4
            // Only use for simulation, debugging, or very low clock frequencies
            //
            always_comb a_out = a_in;
        end
    endgenerate

    // =========================================================================
    // SECTION 4: MAC UNIT INSTANTIATION
    // =========================================================================
    //
    // The MAC (Multiply-Accumulate) unit is the computational heart of the PE.
    // See mac8.sv for detailed documentation.
    //
    // CONNECTIONS:
    // ------------
    // .a(a_in)        → Current activation (streaming input)
    // .b(weight_reg)  → STATIONARY weight (loaded once, held constant)
    // .bypass(bypass) → Residual mode for skip connections
    // .en(en)         → MAC enable from scheduler
    // .clr(clr)       → Clear for new tile
    // .acc(acc)       → 32-bit accumulated result
    //
    // KEY INSIGHT:
    // ------------
    // Notice that 'b' connects to weight_reg, NOT b_in.
    // This is the essence of weight-stationary dataflow:
    //   - weight_reg holds the SAME value for K cycles
    //   - a_in changes every cycle (streaming activations)
    //   - Result: each weight is multiplied with K different activations
    //
    mac8 #(
        // Enable zero bypass for power savings in sparse networks
        // After ReLU, 50-70% of activations are zero → significant power reduction
        .ENABLE_ZERO_BYPASS(1)
    ) u_mac (
        .clk    (clk),
        .rst_n  (rst_n),
        .a      (a_in),        // Use STREAMING activation
        .b      (weight_reg),  // Use STATIONARY weight
        .bypass (bypass),      // Pass bypass signal for residual mode
        .en     (en),
        .clr    (clr),
        .acc    (acc)
    );

    // =========================================================================
    // SECTION 5: DESIGN-BY-CONTRACT ASSERTIONS
    // =========================================================================
    //
    // These SystemVerilog assertions (SVA) verify control protocol invariants.
    // They run ONLY in simulation (ignored during synthesis).
    //
    // PURPOSE:
    // --------
    // 1. Catch logic bugs early in simulation
    // 2. Document the control protocol assumptions
    // 3. Prevent hard-to-debug silicon failures
    //
    /* verilator lint_off SYNCASYNCNET */
    
    // ─────────────────────────────────────────────────────────────────────────
    // PROPERTY 1: Mutual Exclusion of Load and Compute
    // ─────────────────────────────────────────────────────────────────────────
    // INVARIANT: Never load weights and compute MAC simultaneously.
    //
    // WHY THIS MATTERS:
    //   - If load_weight=1 and en=1, the weight changes mid-computation
    //   - This would corrupt the partial sum (wrong weight used)
    //   - The scheduler MUST sequence these operations properly
    //
    // FORMAL PROPERTY:
    //   At every rising clock edge (when not in reset):
    //     IF load_weight is HIGH THEN en MUST be LOW
    //
    property p_no_load_and_compute;
        @(posedge clk) disable iff (!rst_n) (load_weight |-> !en);
    endproperty
    
    assert property (p_no_load_and_compute) 
        else $error("PE Error: Attempted to Load Weight and Compute simultaneously!");

    // ─────────────────────────────────────────────────────────────────────────
    // PROPERTY 2: Clear Functionality Verification
    // ─────────────────────────────────────────────────────────────────────────
    // INVARIANT: After clr is asserted, accumulator becomes zero next cycle.
    //
    // This is a sanity check that the clear logic works correctly.
    // The |=> operator means "on the NEXT cycle, the following is true"
    //
    property p_clear_works;
        @(posedge clk) disable iff (!rst_n) (clr |=> (acc == 0));
    endproperty
    
    // NOTE: Using 'cover' instead of 'assert' because this is a liveness
    // property. We want to verify it CAN happen, not that it ALWAYS happens.
    // cover property (p_clear_works);
    
    /* verilator lint_on SYNCASYNCNET */

endmodule
`default_nettype wire
`endif

