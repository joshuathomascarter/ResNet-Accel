// =============================================================================
// perf.sv — Non-Intrusive Performance Monitor for Systolic Array Accelerator
// =============================================================================
//
// DESCRIPTION:
// ------------
// Measures execution time and utilization of the accelerator by observing
// control signals and counting clock cycles. Provides data for performance
// analysis, optimization, and debugging.
//
// The monitor is non-intrusive: it only observes signals, never modifies them.
// This ensures accurate measurements without affecting accelerator behavior.
//
// MEASURED METRICS:
// -----------------
// 1. total_cycles_count: Clock cycles from start_pulse to done_pulse
//    - Includes all phases: DMA, compute, stalls
//    - Use for end-to-end latency measurement
//
// 2. active_cycles_count: Cycles with busy_signal=1
//    - Represents cycles doing useful work
//    - Utilization = active_cycles / total_cycles
//
// 3. idle_cycles_count: Cycles with busy_signal=0 during measurement
//    - Represents stalls, memory waits, pipeline bubbles
//    - idle_cycles + active_cycles = total_cycles
//
// 4. cache_hit_count: Metadata cache hits (from meta_decode)
//    - Indicates cache effectiveness
//    - High hit rate → low memory pressure
//
// 5. cache_miss_count: Metadata cache misses (from meta_decode)
//    - Each miss adds 2+ cycles latency
//    - Optimize: Sequential row access pattern
//
// 6. decode_count: Total metadata decode operations
//    - One per BSR block processed
//    - Matches nnz_blocks for correct operation
//
// FSM STATE DIAGRAM:
// ------------------
//   ┌───────────────────────────────────┐
//   │                                   │
//   │   ┌────────┐   start_pulse   ┌────┴─────┐
//   └──►│ S_IDLE │────────────────►│S_MEASURING│
//       └────────┘                 └─────┬─────┘
//            ▲                           │
//            │       done_pulse          │
//            └───────────────────────────┘
//
// TIMING:
// -------
//   Cycle 0:   start_pulse asserted
//   Cycle 1:   state_reg = S_MEASURING, counters start
//   Cycle N:   done_pulse asserted
//   Cycle N+1: state_reg = S_IDLE, counters latched to outputs
//              measurement_done asserted for 1 cycle
//
// USAGE EXAMPLE:
// --------------
//   1. Configure accelerator via CSR
//   2. Trigger computation (start_pulse)
//   3. Wait for done (poll STATUS or use interrupt)
//   4. Read PERF_TOTAL_CYCLES, PERF_ACTIVE_CYCLES via CSR
//   5. Compute: Throughput = (M × N × K × 2 ops) / total_cycles
//
// OVERFLOW BEHAVIOR:
// ------------------
// 32-bit counters wrap at 2^32 cycles = 42.9 seconds @ 100 MHz.
// For longer measurements, software must handle wraparound or
// use COUNTER_WIDTH = 64.
//
// RESOURCE ESTIMATES:
// -------------------
//   - FFs: 6 × COUNTER_WIDTH + ~10 = ~200 for 32-bit
//   - LUTs: ~50 (FSM, counter control)
//   - No BRAM (all register-based)
//
// =============================================================================

`default_nettype none

module perf #(
    // =========================================================================
    // PARAMETERS
    // =========================================================================
    // COUNTER_WIDTH: Bit width of all performance counters
    // - 32 bits: Max 4.29 billion cycles = 42.9 sec @ 100 MHz
    // - 64 bits: Virtually unlimited (584 years @ 100 MHz)
    // - Trade-off: 32-bit saves ~200 FFs vs 64-bit
    parameter COUNTER_WIDTH = 32
)(
    // =========================================================================
    // SYSTEM INPUTS
    // =========================================================================
    input  wire clk,
    input  wire rst_n,

    // =========================================================================
    // CONTROL INPUTS (from accelerator core)
    // =========================================================================
    // start_pulse: Single-cycle pulse to begin measurement
    // - Typically connected to CSR start_pulse output
    // - Clears all internal counters and starts measurement
    input  wire start_pulse,
    
    // done_pulse: Single-cycle pulse to stop measurement
    // - Typically connected to scheduler done output
    // - Latches counter values to outputs
    input  wire done_pulse,
    
    // busy_signal: High when accelerator is doing useful work
    // - Typically: act_dma_busy | bsr_dma_busy | sched_busy
    // - Used to distinguish active vs idle cycles
    input  wire busy_signal,

    // =========================================================================
    // METADATA CACHE INPUTS (from meta_decode)
    // =========================================================================
    // These inputs are sampled at done_pulse time and latched to outputs.
    // Connect to meta_decode's internal counters if available, else tie to 0.
    input  wire [COUNTER_WIDTH-1:0] meta_cache_hits,    // Cache hit count
    input  wire [COUNTER_WIDTH-1:0] meta_cache_misses,  // Cache miss count
    input  wire [COUNTER_WIDTH-1:0] meta_decode_cycles, // Total decode operations

    // =========================================================================
    // STATUS OUTPUTS (mapped to CSRs)
    // =========================================================================
    // All outputs are latched at measurement end and held until next start.
    // Software reads these via CSR registers (PERF_TOTAL_CYCLES, etc.)
    output reg [COUNTER_WIDTH-1:0] total_cycles_count,  // Start-to-done cycles
    output reg [COUNTER_WIDTH-1:0] active_cycles_count, // busy_signal=1 cycles
    output reg [COUNTER_WIDTH-1:0] idle_cycles_count,   // busy_signal=0 cycles
    output reg [COUNTER_WIDTH-1:0] cache_hit_count,     // Metadata cache hits
    output reg [COUNTER_WIDTH-1:0] cache_miss_count,    // Metadata cache misses
    output reg [COUNTER_WIDTH-1:0] decode_count,        // Decode operations
    
    // measurement_done: Single-cycle pulse when measurement complete
    // - Can be used to trigger interrupt or polling check
    output reg                     measurement_done
);

    // =========================================================================
    // FSM STATE DEFINITIONS
    // =========================================================================
    // Simple 2-state FSM: IDLE (waiting) and MEASURING (counting)
    localparam S_IDLE      = 1'b0;
    localparam S_MEASURING = 1'b1;

    reg state_reg, state_next;
    reg prev_state;  // For edge detection on state transitions

    // =========================================================================
    // INTERNAL COUNTERS
    // =========================================================================
    // These counters increment during measurement and are latched to outputs
    // when measurement ends. This prevents glitchy output values during
    // active measurement.
    reg [COUNTER_WIDTH-1:0] total_counter;   // Increments every cycle
    reg [COUNTER_WIDTH-1:0] active_counter;  // Increments when busy_signal=1
    reg [COUNTER_WIDTH-1:0] idle_counter;    // Increments when busy_signal=0

    // =========================================================================
    // FSM STATE REGISTER
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_reg  <= S_IDLE;
            prev_state <= S_IDLE;
        end else begin
            prev_state <= state_reg;  // Save for edge detection
            state_reg  <= state_next;
        end
    end

    // =========================================================================
    // FSM NEXT STATE LOGIC
    // =========================================================================
    always @(*) begin
        state_next = state_reg;  // Default: stay in current state
        
        case (state_reg)
            S_IDLE: begin
                // Start measurement on start_pulse
                if (start_pulse) begin
                    state_next = S_MEASURING;
                end
            end
            
            S_MEASURING: begin
                // Stop measurement on done_pulse
                if (done_pulse) begin
                    state_next = S_IDLE;
                end
            end
        endcase
    end

    // =========================================================================
    // COUNTER LOGIC
    // =========================================================================
    // Manages internal counters during measurement phase.
    // Counters are cleared at measurement start and increment during
    // S_MEASURING state.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_counter  <= {COUNTER_WIDTH{1'b0}};
            active_counter <= {COUNTER_WIDTH{1'b0}};
            idle_counter   <= {COUNTER_WIDTH{1'b0}};
        end else begin
            // Clear counters on measurement start (IDLE → MEASURING transition)
            if (state_reg == S_IDLE && state_next == S_MEASURING) begin
                total_counter  <= {COUNTER_WIDTH{1'b0}};
                active_counter <= {COUNTER_WIDTH{1'b0}};
                idle_counter   <= {COUNTER_WIDTH{1'b0}};
            end
            // Increment counters during measurement
            else if (state_reg == S_MEASURING) begin
                total_counter <= total_counter + 1'b1;
                
                if (busy_signal) begin
                    active_counter <= active_counter + 1'b1;
                end else begin
                    idle_counter <= idle_counter + 1'b1;
                end
            end
        end
    end

    // =========================================================================
    // OUTPUT LATCH LOGIC
    // =========================================================================
    // Latches final counter values to output registers when measurement ends.
    // This provides stable values for software to read via CSR.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            total_cycles_count  <= {COUNTER_WIDTH{1'b0}};
            active_cycles_count <= {COUNTER_WIDTH{1'b0}};
            idle_cycles_count   <= {COUNTER_WIDTH{1'b0}};
            cache_hit_count     <= {COUNTER_WIDTH{1'b0}};
            cache_miss_count    <= {COUNTER_WIDTH{1'b0}};
            decode_count        <= {COUNTER_WIDTH{1'b0}};
            measurement_done    <= 1'b0;
        end else begin
            // Latch final counts on MEASURING → IDLE transition
            if (prev_state == S_MEASURING && state_reg == S_IDLE) begin
                total_cycles_count  <= total_counter;
                active_cycles_count <= active_counter;
                idle_cycles_count   <= idle_counter;
                cache_hit_count     <= meta_cache_hits;
                cache_miss_count    <= meta_cache_misses;
                decode_count        <= meta_decode_cycles;
                measurement_done    <= 1'b1;  // 1-cycle pulse
            end else begin
                measurement_done <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
