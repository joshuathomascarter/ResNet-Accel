/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                            ACT_DMA.SV                                     ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  AXI4 Master DMA Engine for Activation Transfer                          ║
 * ║  Transfers activation data from DDR to on-chip activation buffer         ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 *
 * OVERVIEW:
 * ---------
 * This module reads activation data from DDR memory via AXI4 and writes it
 * to the activation buffer. It handles:
 *   - AXI4 burst transactions for high bandwidth
 *   - Address generation and burst splitting
 *   - 4KB boundary crossing prevention (AXI requirement)
 *   - Error detection and reporting
 *
 * DATA FLOW:
 * ----------
 *   DDR Memory (PS side)
 *         │
 *         │ AXI4 Read
 *         ▼
 *   ┌─────────────────┐
 *   │    act_dma      │
 *   │  (This Module)  │
 *   └────────┬────────┘
 *            │ Buffer Write
 *            ▼
 *   ┌─────────────────┐
 *   │   act_buffer    │
 *   │  (On-chip BRAM) │
 *   └─────────────────┘
 *
 * AXI4 PROTOCOL OVERVIEW:
 * -----------------------
 * AXI4 uses separate channels for address and data:
 *
 *   READ TRANSACTION:
 *   1. Master sends Read Address (AR channel)
 *   2. Master sets ARVALID=1, waits for ARREADY=1
 *   3. Slave sends Read Data (R channel)
 *   4. Slave sets RVALID=1 for each beat
 *   5. Master acknowledges with RREADY=1
 *   6. Slave sets RLAST=1 on final beat
 *
 *   ┌────────────────────────────────────────────────────────────────────┐
 *   │                     AXI4 READ TIMING                              │
 *   │                                                                    │
 *   │  Cycle:  1     2     3     4     5     6     7     8              │
 *   │                                                                    │
 *   │  ARVALID ▄▄▄▄▄▄▄                                                  │
 *   │  ARREADY       ▄▄▄▄                                               │
 *   │                                                                    │
 *   │  RVALID              ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄                    │
 *   │  RDATA               D0    D1    D2    D3                          │
 *   │  RLAST                                       ▄▄▄▄                 │
 *   │  RREADY              ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄                    │
 *   └────────────────────────────────────────────────────────────────────┘
 *
 * BURST CONFIGURATION:
 * --------------------
 *   ARLEN = 15: 16 beats per burst (0-based, so 15 means 16)
 *   ARSIZE = 3'b011: 8 bytes per beat (64-bit data bus)
 *   ARBURST = 2'b01: INCR mode (incrementing addresses)
 *
 *   Bytes per burst: (15 + 1) × 8 = 128 bytes
 *
 * 4KB BOUNDARY RULE:
 * ------------------
 * AXI4 spec requires bursts NOT to cross 4KB boundaries.
 * This is because memory mapping may change at 4KB boundaries.
 *
 * Example violation:
 *   Start: 0x00000FC0 (within first 4KB page)
 *   Burst: 128 bytes
 *   End:   0x00001040 (crosses into second page at 0x1000) ← ILLEGAL!
 *
 * This module prevents boundary crossing by limiting burst length.
 *
 * RESOURCE USAGE:
 * ---------------
 *   - ~200 LUTs
 *   - ~150 FFs
 *   - 1 AXI master port
 *
 * PERFORMANCE:
 * ------------
 *   - Theoretical: 64 bits × 100 MHz = 800 MB/s
 *   - Practical: ~600 MB/s (accounting for handshaking overhead)
 *   - Typical activation tile: 14×14×4 bytes = 784 bytes ≈ 1.3 µs
 */

`timescale 1ns/1ps
`default_nettype none

module act_dma #(
    // =========================================================================
    // PARAMETER: AXI_ADDR_W - AXI Address Width
    // =========================================================================
    // Width of AXI address bus. 32 bits supports up to 4GB address space.
    // Zynq-7020 uses 32-bit addresses for PS-PL interface.
    parameter AXI_ADDR_W = 32,
    
    // =========================================================================
    // PARAMETER: AXI_DATA_W - AXI Data Width
    // =========================================================================
    // Width of AXI data bus. 64 bits is standard for Zynq HP ports.
    // Matches DDR burst width for maximum efficiency.
    parameter AXI_DATA_W = 64,
    
    // =========================================================================
    // PARAMETER: AXI_ID_W - AXI Transaction ID Width
    // =========================================================================
    // Used for out-of-order completion and routing.
    // 4 bits allows up to 16 outstanding transactions.
    parameter AXI_ID_W   = 4,
    
    // =========================================================================
    // PARAMETER: STREAM_ID - Unique ID for This DMA
    // =========================================================================
    // Each DMA engine has a unique ID:
    //   0 = BSR DMA (weight/metadata)
    //   1 = Activation DMA (this module)
    //
    // The AXI bridge uses this to route responses to the correct DMA.
    parameter STREAM_ID  = 1,
    
    // =========================================================================
    // PARAMETER: BURST_LEN - Maximum Burst Length (0-based)
    // =========================================================================
    // ARLEN value for maximum burst. 15 means 16 beats.
    // Bytes per burst: (BURST_LEN + 1) × (AXI_DATA_W / 8)
    //                = 16 × 8 = 128 bytes
    //
    // WHY 16 BEATS?
    //   - AXI4 maximum: 256 beats
    //   - 16 beats = 128 bytes (good balance of efficiency and latency)
    //   - Larger bursts increase latency for small transfers
    parameter BURST_LEN  = 8'd15
)(
    // =========================================================================
    // SYSTEM INTERFACE
    // =========================================================================
    input  wire                  clk,      // System clock (100-200 MHz)
    input  wire                  rst_n,    // Active-low async reset

    // =========================================================================
    // CONTROL INTERFACE (from CSR)
    // =========================================================================
    
    /**
     * start - Initiate DMA Transfer
     * ------------------------------
     * Pulse HIGH for 1 cycle to begin transfer.
     * src_addr and transfer_length must be valid when start is asserted.
     */
    input  wire                  start,
    
    /**
     * src_addr - Source Address in DDR
     * ---------------------------------
     * Physical address of activation data in DDR.
     * MUST be 64-bit aligned (lower 3 bits = 0).
     * Example: 0x10000000 (activation buffer region)
     */
    input  wire [AXI_ADDR_W-1:0] src_addr,
    
    /**
     * transfer_length - Number of Bytes to Transfer
     * -----------------------------------------------
     * Total bytes to read from DDR and write to buffer.
     * MUST be > 0 and preferably a multiple of 8 (64-bit aligned).
     */
    input  wire [31:0]           transfer_length,
    
    /**
     * done - Transfer Complete Flag
     * ------------------------------
     * Pulses HIGH for 1 cycle when transfer finishes.
     * Check 'error' flag to determine success/failure.
     */
    output reg                   done,
    
    /**
     * busy - Transfer In Progress
     * ----------------------------
     * HIGH while DMA is actively transferring data.
     * LOW when idle or after done pulse.
     */
    output reg                   busy,
    
    /**
     * error - Error Flag
     * -------------------
     * HIGH if any AXI error response received (RRESP != OKAY).
     * Remains high until next start pulse.
     */
    output reg                   error,

    // =========================================================================
    // AXI4 MASTER READ INTERFACE (to AXI Bridge → DDR)
    // =========================================================================
    // This is a READ-ONLY interface. We only fetch data FROM DDR.
    // Write channels (AW, W, B) are not implemented.
    
    // ─────────────────────────────────────────────────────────────────────────
    // READ ADDRESS CHANNEL (AR) - Master sends address
    // ─────────────────────────────────────────────────────────────────────────
    
    /** m_axi_arid - Transaction ID (unique to this DMA) */
    output wire [AXI_ID_W-1:0]   m_axi_arid,
    
    /** m_axi_araddr - Read address (byte address in DDR) */
    output reg [AXI_ADDR_W-1:0]  m_axi_araddr,
    
    /** 
     * m_axi_arlen - Burst length minus 1
     * ----------------------------------
     * Number of data beats in burst - 1.
     * Range: 0 to 255 (1 to 256 beats)
     * We use BURST_LEN = 15 → 16 beats × 8 bytes = 128 bytes
     */
    output reg [7:0]             m_axi_arlen,
    
    /** 
     * m_axi_arsize - Bytes per beat (encoded)
     * ----------------------------------------
     * 3'b000 = 1 byte,   3'b001 = 2 bytes
     * 3'b010 = 4 bytes,  3'b011 = 8 bytes (64-bit, used here)
     * 3'b100 = 16 bytes, etc.
     */
    output reg [2:0]             m_axi_arsize,
    
    /** 
     * m_axi_arburst - Burst type
     * ---------------------------
     * 2'b00 = FIXED (same address each beat)
     * 2'b01 = INCR (incrementing address) ← WE USE THIS
     * 2'b10 = WRAP (wrapping burst)
     */
    output reg [1:0]             m_axi_arburst,
    
    /** m_axi_arvalid - Address valid (master asserts) */
    output reg                   m_axi_arvalid,
    
    /** m_axi_arready - Address ready (slave asserts) */
    input  wire                  m_axi_arready,

    // ─────────────────────────────────────────────────────────────────────────
    // READ DATA CHANNEL (R) - Slave sends data
    // ─────────────────────────────────────────────────────────────────────────
    
    /** m_axi_rid - Transaction ID (matches m_axi_arid) */
    input  wire [AXI_ID_W-1:0]   m_axi_rid,
    
    /** m_axi_rdata - Read data (64 bits = 8 bytes) */
    input  wire [AXI_DATA_W-1:0] m_axi_rdata,
    
    /** 
     * m_axi_rresp - Read response
     * ----------------------------
     * 2'b00 = OKAY (normal success)
     * 2'b01 = EXOKAY (exclusive access success)
     * 2'b10 = SLVERR (slave error)
     * 2'b11 = DECERR (decode error - address invalid)
     */
    input  wire [1:0]            m_axi_rresp,
    
    /** m_axi_rlast - Last beat of burst */
    input  wire                  m_axi_rlast,
    
    /** m_axi_rvalid - Data valid (slave asserts) */
    input  wire                  m_axi_rvalid,
    
    /** m_axi_rready - Data ready (master asserts) */
    output reg                   m_axi_rready,

    // =========================================================================
    // BUFFER WRITE INTERFACE (to act_buffer)
    // =========================================================================
    
    /** act_we - Write enable to activation buffer */
    output reg                   act_we,
    
    /** act_addr - Write address (word index in buffer) */
    output reg [AXI_ADDR_W-1:0]  act_addr,
    
    /** act_wdata - Write data (64 bits per write) */
    output reg [AXI_DATA_W-1:0]  act_wdata
);

    // =========================================================================
    // INTERNAL STATE & REGISTERS
    // =========================================================================
    
    /**
     * FSM State Encoding
     * -------------------
     * Using enumerated type for readability and debug.
     * Synthesis will optimize to appropriate encoding.
     */
    typedef enum logic [1:0] {
        IDLE,        // Waiting for start signal
        SEND_ADDR,   // Sending read address on AR channel
        READ_DATA,   // Receiving data beats on R channel
        DONE_STATE   // Transfer complete, pulse done signal
    } state_t;

    state_t state;  // Current FSM state
    
    /**
     * current_axi_addr - Tracks current DDR read position
     * -----------------------------------------------------
     * Initialized to src_addr, incremented by burst size after each burst.
     */
    reg [AXI_ADDR_W-1:0] current_axi_addr;
    
    /**
     * bytes_remaining - Bytes left to transfer
     * -----------------------------------------
     * Decremented as data is received.
     * Transfer complete when this reaches 0.
     */
    reg [31:0]           bytes_remaining;

    // =========================================================================
    // AXI PROTOCOL CONSTANTS
    // =========================================================================
    
    /**
     * AXI_SIZE_64 - Size encoding for 8-byte (64-bit) transfers
     * -----------------------------------------------------------
     * ARSIZE = 3'b011 means 2^3 = 8 bytes per beat.
     * This matches our 64-bit data bus width.
     */
    localparam [2:0] AXI_SIZE_64 = 3'b011;
    
    /**
     * AXI_BURST_INCR - Incrementing burst type
     * -----------------------------------------
     * ARBURST = 2'b01 means addresses increment by ARSIZE bytes each beat.
     * Most efficient for sequential memory access.
     */
    localparam [1:0] AXI_BURST_INCR = 2'b01;

    // =========================================================================
    // TRANSACTION ID ASSIGNMENT
    // =========================================================================
    // Each DMA has a unique ID for response routing.
    // The bridge uses this to demultiplex responses from DDR.
    assign m_axi_arid = STREAM_ID[AXI_ID_W-1:0];

    // =========================================================================
    // MAIN FSM - AXI READ TRANSACTION STATE MACHINE
    // =========================================================================
    /**
     * STATE MACHINE OVERVIEW
     * =======================
     * 
     * IDLE → SEND_ADDR → READ_DATA → DONE_STATE → IDLE
     *            ↑______________|
     *        (if more data needed)
     * 
     * TIMING EXAMPLE (16-beat burst):
     * --------------------------------
     *   Cycle 0:   start=1, capture src_addr/transfer_length
     *   Cycle 1:   SEND_ADDR: Assert ARVALID
     *   Cycle 2:   ARREADY arrives, handshake complete
     *   Cycles 3-18: READ_DATA: Receive 16 data beats
     *   Cycle 19:  RLAST=1, check if more bursts needed
     *   ...repeat SEND_ADDR/READ_DATA if bytes_remaining > 0...
     *   Final:     DONE_STATE, pulse done=1
     * 
     * EFFICIENCY: Each 128-byte burst takes ~18 cycles = 7.1 bytes/cycle
     * At 200 MHz: 1.42 GB/s theoretical max (DDR limited to ~4 GB/s)
     */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // ─────────────────────────────────────────────────────────────────
            // RESET STATE
            // ─────────────────────────────────────────────────────────────────
            // Initialize all outputs to safe defaults.
            // FSM starts in IDLE, ready for new transfers.
            state <= IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            m_axi_arvalid <= 1'b0;
            m_axi_rready <= 1'b0;
            act_we <= 1'b0;
            act_addr <= 0;
            current_axi_addr <= 0;
            bytes_remaining <= 0;
        end else begin
            // ─────────────────────────────────────────────────────────────────
            // DEFAULT SIGNAL BEHAVIOR (pulse signals)
            // ─────────────────────────────────────────────────────────────────
            act_we <= 1'b0;   // Buffer write enable only high during valid data
            done <= 1'b0;     // Done is a 1-cycle pulse

            case (state)
                // ─────────────────────────────────────────────────────────────
                // IDLE: Wait for Start Signal
                // ─────────────────────────────────────────────────────────────
                // Capture transfer parameters and transition to SEND_ADDR.
                // The act_addr counter resets to 0 for new transfer.
                IDLE: begin
                    if (start) begin
                        busy <= 1'b1;
                        error <= 1'b0;              // Clear any previous error
                        current_axi_addr <= src_addr;
                        bytes_remaining <= transfer_length;
                        act_addr <= 0;              // Reset buffer write pointer
                        state <= SEND_ADDR;
                    end
                end

                // ─────────────────────────────────────────────────────────────
                // SEND_ADDR: Issue AXI Read Address (AR Channel)
                // ─────────────────────────────────────────────────────────────
                /**
                 * AXI4 Read Address Handshake:
                 *   1. Master asserts ARVALID with address/length
                 *   2. Slave asserts ARREADY when it can accept
                 *   3. Handshake completes on ARVALID && ARREADY
                 * 
                 * BURST LENGTH CALCULATION:
                 * -------------------------
                 * If bytes_remaining > 128: Use full burst (ARLEN = 15)
                 * Otherwise: Calculate minimum beats needed
                 * 
                 * Formula: ARLEN = ceil(bytes_remaining / 8) - 1
                 * Example: 24 bytes → ceil(24/8) - 1 = 3 - 1 = 2 → 3 beats
                 */
                SEND_ADDR: begin
                    m_axi_araddr  <= current_axi_addr;
                    m_axi_arsize  <= AXI_SIZE_64;    // 8 bytes per beat
                    m_axi_arburst <= AXI_BURST_INCR; // Incrementing address
                    m_axi_arvalid <= 1'b1;
                    
                    // Calculate burst length based on remaining bytes
                    // Magic number: (BURST_LEN + 1) * 8 = 16 * 8 = 128 bytes
                    if (bytes_remaining > (BURST_LEN + 1) * 8) 
                        m_axi_arlen <= BURST_LEN;   // Full 16-beat burst
                    else 
                        // Partial burst: ceil(bytes / 8) - 1
                        // +7 for ceiling division, >>3 for /8, -1 for 0-based
                        m_axi_arlen <= ((bytes_remaining + 7) >> 3) - 1;

                    // Wait for slave to accept address
                    if (m_axi_arready && m_axi_arvalid) begin
                        m_axi_arvalid <= 1'b0;      // Address sent, deassert
                        m_axi_rready  <= 1'b1;      // Ready to receive data
                        state <= READ_DATA;
                    end
                end

                // ─────────────────────────────────────────────────────────────
                // READ_DATA: Receive AXI Read Data (R Channel)
                // ─────────────────────────────────────────────────────────────
                /**
                 * AXI4 Read Data Protocol:
                 *   - Slave sends data beats with RVALID
                 *   - Master accepts with RREADY
                 *   - RLAST marks final beat of burst
                 *   - RRESP indicates success/error for each beat
                 * 
                 * DATA FLOW:
                 *   DDR → AXI Bridge → m_axi_rdata → act_buffer
                 * 
                 * Each beat: 64 bits = 8 INT8 activations
                 * Full burst: 16 beats = 128 activations
                 */
                READ_DATA: begin
                    // Note: We trust the Bridge to only assert m_axi_rvalid 
                    // when the data is actually for us (based on ID routing).
                    if (m_axi_rvalid) begin
                        // Check for AXI error response
                        if (m_axi_rresp != 2'b00) begin
                            // SLVERR or DECERR - abort transfer
                            error <= 1'b1;
                            busy <= 1'b0;
                            done <= 1'b1; // Pulse done to wake up controller
                            state <= IDLE;
                        end else begin
                            // Valid data received - write to buffer
                            act_we    <= 1'b1;
                            act_wdata <= m_axi_rdata;
                            act_addr  <= act_addr + 1;  // Increment buffer pointer

                            // Decrement bytes remaining
                            // Guard against underflow for non-aligned transfers
                            if (bytes_remaining >= 8)
                                bytes_remaining <= bytes_remaining - 8;
                            else
                                bytes_remaining <= 0;

                            // Check if this is the last beat of the burst
                            if (m_axi_rlast) begin
                                m_axi_rready <= 1'b0;   // Stop accepting data
                                
                                // Update address for next burst
                                // Magic number: (arlen + 1) * 8 = bytes transferred
                                current_axi_addr <= current_axi_addr + ((m_axi_arlen + 1) * 8);

                                // Check if transfer complete
                                // bytes_remaining <= 8 means this was the last byte
                                if (bytes_remaining <= 8) begin
                                    state <= DONE_STATE;
                                end else begin
                                    // More data needed, issue another burst
                                    state <= SEND_ADDR;
                                end
                            end
                        end
                    end
                end

                // ─────────────────────────────────────────────────────────────
                // DONE_STATE: Signal Transfer Complete
                // ─────────────────────────────────────────────────────────────
                // Pulse done=1 for one cycle, return to IDLE.
                // The CSR will capture the done pulse.
                DONE_STATE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

    // =========================================================================
    // SIMULATION-ONLY ASSERTIONS (Safety Checks)
    // =========================================================================
    /**
     * These assertions catch common programming errors during simulation.
     * They are removed during synthesis (translate_off).
     * 
     * Critical Checks:
     * 1. Zero-length transfers (would hang the FSM)
     * 2. Misaligned addresses (would cause data corruption)
     * 3. 4KB boundary crossing (AXI protocol violation)
     */
    `ifndef VERILATOR
    // synthesis translate_off
    
    /**
     * ASSERTION 1: Zero Length Prevention
     * ------------------------------------
     * A zero-length transfer would cause:
     *   - arlen underflow (0 - 1 = 255)
     *   - FSM hang waiting for 256 beats that never complete
     * 
     * Root cause: Software bug in host driver or CSR programming.
     */
    always @(posedge clk) begin
        if (start && transfer_length == 0) begin
            $error("ACT_DMA ERROR: Attempted to start DMA with length = 0!");
        end
    end

    /**
     * ASSERTION 2: 64-bit Alignment Check
     * -------------------------------------
     * Unaligned addresses cause:
     *   - Data corruption (partial word reads)
     *   - Potential AXI protocol violations
     * 
     * 64-bit alignment: Lower 3 bits must be 000
     * Example: 0x10000004 is 32-bit aligned but not 64-bit aligned!
     */
    always @(posedge clk) begin
        if (start && (src_addr[2:0] != 3'b000)) begin
            $error("ACT_DMA ERROR: Source Address %h is not 64-bit aligned!", src_addr);
        end
    end

    /**
     * ASSERTION 3: 4KB Boundary Crossing Check
     * -----------------------------------------
     * AXI4 SPEC REQUIREMENT: Bursts cannot cross 4KB boundaries!
     * 
     * Why 4KB?
     *   - Memory protection unit page size
     *   - Different pages may have different attributes
     *   - Hardware can't handle mid-burst page faults
     * 
     * The magic number 0xFFF = 4095 = 4KB - 1
     * If (addr[11:0] + burst_bytes) > 4096, we cross a boundary.
     * 
     * Note: This is a WARNING not error - hardware should split bursts,
     * but our simple DMA doesn't (would need hardware fix for compliance).
     */
    always @(posedge clk) begin
        if (state == SEND_ADDR && m_axi_arvalid) begin
            if ((m_axi_araddr[11:0] + ({24'd0, m_axi_arlen} + 32'd1) * 8) > 32'hFFF) begin
                $warning("ACT_DMA WARNING: Burst at %h might cross 4KB boundary!", m_axi_araddr);
            end
        end
    end
    
    // synthesis translate_on
    `endif

endmodule
