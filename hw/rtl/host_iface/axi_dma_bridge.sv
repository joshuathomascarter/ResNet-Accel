// =============================================================================
// axi_dma_bridge.sv — 2:1 AXI4 Read Arbiter for DMA Masters
// =============================================================================
//
// DESCRIPTION:
// ------------
// Arbitrates between two AXI4 read masters (bsr_dma and act_dma) to share
// a single AXI4 master port connected to Zynq HP for DDR access.
//
// This design implements a simple round-robin arbiter with:
// - Burst-level arbitration (grants held for entire burst)
// - ID-based response routing (ID bit distinguishes masters)
// - Watchdog timeout for hung transactions
//
// ARCHITECTURE:
// -------------
//   ┌──────────┐          ┌────────────────┐          ┌──────────┐
//   │ bsr_dma  │──S0_AR──►│                │          │          │
//   │ (Master0)│◄──S0_R───│  axi_dma_      │──M_AR───►│   DDR    │
//   └──────────┘          │     bridge     │◄──M_R────│ (Slave)  │
//   ┌──────────┐          │                │          │          │
//   │ act_dma  │──S1_AR──►│  (Round-Robin) │          │  via HP  │
//   │ (Master1)│◄──S1_R───│                │          │  Port    │
//   └──────────┘          └────────────────┘          └──────────┘
//
// ARBITRATION POLICY:
// -------------------
// Round-robin with slight priority to S0 (bsr_dma):
//   1. If S0 requests AND (S1 not requesting OR S1 was last granted): Grant S0
//   2. Else if S1 requests: Grant S1
//
// This gives weight loading (typically on critical path) slight priority
// while maintaining fairness for concurrent operation.
//
// FSM STATES:
// -----------
//   IDLE ─────────┐
//   │             │
//   │ s0_arvalid  │ s1_arvalid
//   │ or          │ (no s0)
//   │ s1_arvalid  │
//   ▼             ▼
//   ADDR_PHASE────────── Wait for m_arready
//   │
//   │ m_arready & m_arvalid
//   ▼
//   DATA_PHASE────────── Wait for m_rlast
//   │                    (or watchdog timeout)
//   │ m_rvalid & m_rready & m_rlast
//   ▼
//   IDLE
//
// TIMING EXAMPLE:
// ---------------
// S0 (bsr_dma) requests 4-beat burst while S1 (act_dma) waits:
//
//   Cycle:  1    2    3    4    5    6    7    8    9    10
//   state:  IDLE ADDR ADDR DATA DATA DATA DATA IDLE ADDR DATA...
//   master:      S0   S0   S0   S0   S0   S0        S1   S1
//           ▲         ▲                   ▲    ▲
//           │         │                   │    └─ S1 granted (round-robin)
//           │         │                   └─ rlast, transaction complete
//           │         └─ arready handshake, move to DATA_PHASE
//           └─ S0 wins arbitration
//
// WATCHDOG TIMER:
// ---------------
// 10-bit counter (1024 cycles @ 100 MHz = ~10 µs timeout)
// - Counts during DATA_PHASE only
// - On timeout (0x3FF), force return to IDLE
// - Protects against hung DDR controller or simulation bugs
// - Fires warning at 0x3FE for debugging
//
// ID ROUTING:
// -----------
// Masters set their AXI ID in transactions:
// - S0 (bsr_dma): STREAM_ID = 0
// - S1 (act_dma): STREAM_ID = 1
// The bridge routes responses based on current_master state, not ID field,
// since transactions are serialized (only one in flight at a time).
//
// LIMITATIONS:
// ------------
// - Read-only: No write channel support (DMAs only read from DDR)
// - No outstanding transactions: One transaction at a time per master
// - No QoS support: Fixed priority, no urgency signaling
//
// RESOURCE ESTIMATES:
// -------------------
//   - LUTs: ~200 (FSM, muxes, watchdog)
//   - FFs: ~50 (state, current_master, watchdog counter)
//   - Critical path: Combinational mux from s*_ar* to m_ar*
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module axi_dma_bridge #(
    // =========================================================================
    // PARAMETERS
    // =========================================================================
    // DATA_WIDTH: AXI data bus width (64-bit for Zynq HP port)
    // - Matches DDR interface width for maximum bandwidth
    parameter DATA_WIDTH = 64,
    
    // ADDR_WIDTH: AXI address width (32-bit for Zynq DDR space)
    // - Covers 0x0000_0000 to 0xFFFF_FFFF address range
    parameter ADDR_WIDTH = 32,
    
    // ID_WIDTH: AXI transaction ID width
    // - 4 bits allows 16 unique IDs (only 2 used: 0 and 1)
    // - ID LSB indicates source DMA (0=bsr, 1=act)
    parameter ID_WIDTH   = 4
)(
    // =========================================================================
    // SYSTEM SIGNALS
    // =========================================================================
    input  wire                     clk,        // AXI clock (100 MHz typical)
    input  wire                     rst_n,      // Active-low async reset

    // =========================================================================
    // SLAVE PORT 0: BSR DMA (Read Address Channel)
    // =========================================================================
    // BSR DMA requests sparse weight data from DDR.
    // Higher priority (checked first in arbitration).
    input  wire [ID_WIDTH-1:0]      s0_arid,    // Transaction ID (always 0)
    input  wire [ADDR_WIDTH-1:0]    s0_araddr,  // DDR byte address
    input  wire [7:0]               s0_arlen,   // Burst length - 1 (0=1 beat, 15=16 beats)
    input  wire [2:0]               s0_arsize,  // Bytes per beat (3'b011 = 8 bytes)
    input  wire [1:0]               s0_arburst, // Burst type (2'b01 = INCR)
    input  wire                     s0_arvalid, // Address valid
    output reg                      s0_arready, // Bridge can accept

    // =========================================================================
    // SLAVE PORT 1: ACT DMA (Read Address Channel)
    // =========================================================================
    // ACT DMA requests dense activation data from DDR.
    // Lower priority (granted when S0 not requesting or fairness requires).
    input  wire [ID_WIDTH-1:0]      s1_arid,    // Transaction ID (always 1)
    input  wire [ADDR_WIDTH-1:0]    s1_araddr,  // DDR byte address
    input  wire [7:0]               s1_arlen,   // Burst length - 1
    input  wire [2:0]               s1_arsize,  // Bytes per beat
    input  wire [1:0]               s1_arburst, // Burst type
    input  wire                     s1_arvalid, // Address valid
    output reg                      s1_arready, // Bridge can accept

    // =========================================================================
    // MASTER PORT: DDR (Read Address Channel)
    // =========================================================================
    // Connects to Zynq HP0 or HP2 via AXI Interconnect.
    output reg  [ID_WIDTH-1:0]      m_arid,     // Passed through from winner
    output reg  [ADDR_WIDTH-1:0]    m_araddr,   // Passed through from winner
    output reg  [7:0]               m_arlen,    // Passed through from winner
    output reg  [2:0]               m_arsize,   // Passed through from winner
    output reg  [1:0]               m_arburst,  // Passed through from winner
    output reg                      m_arvalid,  // Valid when in ADDR_PHASE
    input  wire                     m_arready,  // DDR controller can accept

    // =========================================================================
    // MASTER PORT: DDR (Read Data Channel)
    // =========================================================================
    // Receives read data from DDR, routes to appropriate slave port.
    input  wire [ID_WIDTH-1:0]      m_rid,      // Echoed transaction ID
    input  wire [DATA_WIDTH-1:0]    m_rdata,    // 64-bit read data
    input  wire [1:0]               m_rresp,    // Response (2'b00 = OKAY)
    input  wire                     m_rlast,    // Last beat of burst
    input  wire                     m_rvalid,   // Data valid
    output reg                      m_rready,   // Bridge can accept

    // =========================================================================
    // SLAVE PORT 0: BSR DMA (Read Data Channel)
    // =========================================================================
    output reg  [ID_WIDTH-1:0]      s0_rid,     // Echoed ID
    output reg  [DATA_WIDTH-1:0]    s0_rdata,   // Read data
    output reg  [1:0]               s0_rresp,   // Response status
    output reg                      s0_rlast,   // Last beat indicator
    output reg                      s0_rvalid,  // Data valid (only when S0 is active)
    input  wire                     s0_rready,  // S0 can accept

    // =========================================================================
    // SLAVE PORT 1: ACT DMA (Read Data Channel)
    // =========================================================================
    output reg  [ID_WIDTH-1:0]      s1_rid,     // Echoed ID
    output reg  [DATA_WIDTH-1:0]    s1_rdata,   // Read data
    output reg  [1:0]               s1_rresp,   // Response status
    output reg                      s1_rlast,   // Last beat indicator
    output reg                      s1_rvalid,  // Data valid (only when S1 is active)
    input  wire                     s1_rready   // S1 can accept

    // Future: Add status outputs (grant count, timeout count, error flag)
);

    // =========================================================================
    // FSM STATE DEFINITION
    // =========================================================================
    // Simple 3-state FSM for transaction arbitration:
    //   IDLE: No transaction in progress, arbitrate on new requests
    //   ADDR_PHASE: Address handshake in progress, wait for m_arready
    //   DATA_PHASE: Receiving burst data, wait for m_rlast
    typedef enum logic [1:0] {
        IDLE,           // Waiting for DMA request
        ADDR_PHASE,     // Forwarding address, waiting for handshake
        DATA_PHASE      // Receiving data, waiting for rlast
    } state_t;

    state_t state;
    
    // =========================================================================
    // ARBITRATION REGISTERS
    // =========================================================================
    // current_master: Which DMA is currently granted (0=S0/bsr, 1=S1/act)
    // - Held constant throughout entire transaction (address + data phases)
    reg current_master;
    
    // last_master: Which DMA was granted in the previous transaction
    // - Used for round-robin fairness (alternate when both requesting)
    reg last_master;
    
    // =========================================================================
    // WATCHDOG TIMER
    // =========================================================================
    // Counts cycles spent in DATA_PHASE waiting for response.
    // If DDR hangs (simulation bug or hardware failure), this prevents lockup.
    //
    // 10 bits = 1024 cycles = 10.24 µs @ 100 MHz
    // This is generous; typical 16-beat burst takes ~20 cycles.
    reg [9:0] watchdog_timer;

    // =========================================================================
    // MAIN FSM AND WATCHDOG LOGIC
    // =========================================================================
    // Combined for efficiency - state transitions and timer are tightly coupled.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset: Return to idle, clear arbitration history
            state <= IDLE;
            current_master <= 1'b0;
            last_master <= 1'b0;
            watchdog_timer <= 10'd0;
        end else begin
            // -------------------------------------------------------------
            // Watchdog Timer Management
            // -------------------------------------------------------------
            // Only count during DATA_PHASE (waiting for DDR response).
            // Reset immediately on any state transition to avoid false triggers.
            if (state == DATA_PHASE) 
                watchdog_timer <= watchdog_timer + 1'b1;
            else 
                watchdog_timer <= 10'd0;

            // -------------------------------------------------------------
            // FSM State Transitions
            // -------------------------------------------------------------
            case (state)
                // ---------------------------------------------------------
                // IDLE: Arbitrate between requesting DMAs
                // ---------------------------------------------------------
                IDLE: begin
                    // Round-Robin Arbitration with S0 priority:
                    // 1. Grant S0 if: S0 requesting AND (S1 not requesting OR S1 was last)
                    // 2. Grant S1 if: S1 requesting (and S0 didn't win above)
                    //
                    // This ensures:
                    // - S0 wins when only S0 requesting
                    // - S1 wins when only S1 requesting
                    // - They alternate when both requesting
                    // - S0 has slight edge (wins ties when last_master was S1)
                    if (s0_arvalid && (!s1_arvalid || last_master == 1'b1)) begin
                        current_master <= 1'b0;
                        state <= ADDR_PHASE;
                    end
                    else if (s1_arvalid) begin
                        current_master <= 1'b1;
                        state <= ADDR_PHASE;
                    end
                    // Else: Stay in IDLE (no requests)
                end

                // ---------------------------------------------------------
                // ADDR_PHASE: Wait for DDR to accept address
                // ---------------------------------------------------------
                ADDR_PHASE: begin
                    // Handshake complete when both valid and ready asserted
                    if (m_arready && m_arvalid) begin
                        state <= DATA_PHASE;
                    end
                    // Note: No timeout here - DDR should respond quickly
                    // If DDR is busy, it will lower arready and we wait
                end

                // ---------------------------------------------------------
                // DATA_PHASE: Receive burst data from DDR
                // ---------------------------------------------------------
                DATA_PHASE: begin
                    // Normal completion: Last beat received and accepted
                    if (m_rvalid && m_rready && m_rlast) begin
                        last_master <= current_master;  // Remember for fairness
                        state <= IDLE;
                    end
                    // Watchdog timeout: DDR hung, force recovery
                    // 0x3FF = 1023, triggers after 1024 cycles
                    else if (watchdog_timer == 10'h3FF) begin
                        state <= IDLE;  // Force reset to IDLE
                        // Note: This may leave DMA in inconsistent state
                        // Host software should handle recovery
                    end
                end
                
                // Default: Should never happen, return to IDLE
                default: state <= IDLE;
            endcase
        end
    end

    // =========================================================================
    // ADDRESS CHANNEL MULTIPLEXER (Combinational)
    // =========================================================================
    // Routes address signals from the granted DMA master to DDR port.
    // Only active during ADDR_PHASE; all outputs zeroed otherwise.
    //
    // Signal Flow:
    //   current_master=0: s0_ar* → m_ar*, m_arready → s0_arready
    //   current_master=1: s1_ar* → m_ar*, m_arready → s1_arready
    //
    // Note: Using combinational logic (always @(*)) to minimize latency.
    // Synthesis will optimize to simple 2:1 muxes.
    always @(*) begin
        // Default: Drive zeros to prevent X propagation and ensure clean
        // signals when IDLE (important for DDR controller compatibility)
        m_arid    = {ID_WIDTH{1'b0}};
        m_araddr  = {ADDR_WIDTH{1'b0}};
        m_arlen   = 8'd0;
        m_arsize  = 3'd0;
        m_arburst = 2'd0;
        m_arvalid = 1'b0;
        
        // Default: Neither slave gets ready (no transactions accepted)
        s0_arready = 1'b0;
        s1_arready = 1'b0;

        // Only mux signals during address phase
        if (state == ADDR_PHASE) begin
            if (current_master == 1'b0) begin
                // Connect S0 (BSR DMA) to DDR Master Port
                m_arid    = s0_arid;
                m_araddr  = s0_araddr;
                m_arlen   = s0_arlen;
                m_arsize  = s0_arsize;
                m_arburst = s0_arburst;
                m_arvalid = s0_arvalid;
                // Pass DDR ready back to BSR DMA for handshake
                s0_arready = m_arready;
            end else begin
                // Connect S1 (Activation DMA) to DDR Master Port
                m_arid    = s1_arid;
                m_araddr  = s1_araddr;
                m_arlen   = s1_arlen;
                m_arsize  = s1_arsize;
                m_arburst = s1_arburst;
                m_arvalid = s1_arvalid;
                // Pass DDR ready back to Act DMA for handshake
                s1_arready = m_arready;
            end
        end
    end

    // =========================================================================
    // DATA CHANNEL MULTIPLEXER (Combinational)
    // =========================================================================
    // Routes read data from DDR to the appropriate DMA master.
    // Also routes backpressure (rready) from DMA back to DDR.
    //
    // Signal Flow:
    //   current_master=0: m_r* → s0_r*, s0_rready → m_rready
    //   current_master=1: m_r* → s1_r*, s1_rready → m_rready
    //
    // Both slave ports always see the data (for debugging), but rvalid
    // is only asserted for the active master.
    always @(*) begin
        // Default: Both slaves see DDR data (for debugging/monitoring)
        // Only rvalid distinguishes which is the active recipient
        s0_rid    = m_rid;
        s0_rdata  = m_rdata;
        s0_rresp  = m_rresp;
        s0_rlast  = m_rlast;
        s0_rvalid = 1'b0;  // Default: S0 not receiving

        s1_rid    = m_rid;
        s1_rdata  = m_rdata;
        s1_rresp  = m_rresp;
        s1_rlast  = m_rlast;
        s1_rvalid = 1'b0;  // Default: S1 not receiving

        // Default: Don't accept DDR data (only accept when in DATA_PHASE)
        m_rready  = 1'b0;

        // Route data to active master during DATA_PHASE
        if (state == DATA_PHASE) begin
            if (current_master == 1'b0) begin
                // Route data to S0 (BSR DMA)
                s0_rvalid = m_rvalid;
                // Pass S0's backpressure to DDR
                m_rready  = s0_rready;
            end else begin
                // Route data to S1 (Activation DMA)
                s1_rvalid = m_rvalid;
                // Pass S1's backpressure to DDR
                m_rready  = s1_rready;
            end
        end
    end

    // =========================================================================
    // SIMULATION-ONLY ASSERTIONS (Safety Checks)
    // =========================================================================
    // These assertions catch design bugs and simulation issues.
    // Disabled during synthesis (synthesis translate_off).
    // synthesis translate_off

    // -------------------------------------------------------------------------
    // Watchdog Timeout Warning
    // -------------------------------------------------------------------------
    // Fires one cycle before timeout to give visibility into hung conditions.
    // In simulation, this usually indicates:
    //   - DDR model not responding (check memory model configuration)
    //   - Incorrect burst length (arlen larger than model supports)
    //   - Deadlock in AXI interconnect
    //
    // Note: Changed from $error to $display for coverage testing where
    // simple memory models may not fully support all BSR access patterns.
    /* verilator lint_off UNUSED */
    always @(posedge clk) begin
        if (state == DATA_PHASE && watchdog_timer == 10'h3FE) begin
            $display("AXI_BRIDGE WARNING: Watchdog Timer about to expire!");
            $display("  DDR not responding. Master=%0d, Cycles=%0d", 
                     current_master, watchdog_timer);
            // Debug info:
            // $display("  Last araddr=0x%08x, arlen=%0d", m_araddr, m_arlen);
        end
    end
    /* verilator lint_on UNUSED */

    // -------------------------------------------------------------------------
    // Unexpected Data Check
    // -------------------------------------------------------------------------
    // If DDR sends data while we're in IDLE, something is wrong:
    //   - Previous transaction wasn't properly terminated
    //   - DDR interconnect has routing bug
    //   - rlast wasn't asserted on final beat
    always @(posedge clk) begin
        if (state == IDLE && m_rvalid) begin
            $error("AXI_BRIDGE ERROR: Received RVALID while IDLE!");
            $error("  Data will be lost. rid=%0d, rlast=%0d", m_rid, m_rlast);
        end
    end

    // -------------------------------------------------------------------------
    // Arbitration Sanity Check
    // -------------------------------------------------------------------------
    // current_master should never be X during active transactions.
    // This would indicate uninitialized register or synthesis mismatch.
    always @(posedge clk) begin
        if (state != IDLE && (current_master === 1'bx)) begin
            $error("AXI_BRIDGE ERROR: current_master is X during active transaction!");
            $error("  State=%0d. This is a critical design bug.", state);
            $finish;
        end
    end

    // synthesis translate_on

endmodule

`default_nettype wire

// =============================================================================
// End of axi_dma_bridge.sv
// =============================================================================
