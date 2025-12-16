// =============================================================================
// axi_lite_slave.sv — AXI4-Lite Slave Interface for CSR Access
// =============================================================================
//
// DESCRIPTION:
// -----------
// Converts AXI4-Lite transactions from the Zynq PS (via GP0) into simple
// register read/write strobes for the CSR module. This module handles all
// AXI protocol complexity (handshaking, channel synchronization, responses).
//
// AXI4-LITE PROTOCOL OVERVIEW:
// ----------------------------
// AXI4-Lite is a simplified AXI4 protocol for control registers:
//   - Single-beat transactions only (no bursts)
//   - 32-bit data width (fixed, per AMBA specification)
//   - 5 channels: AW (write addr), W (write data), B (write resp),
//                 AR (read addr), R (read data)
//   - All channels use valid/ready handshaking
//
// WRITE TRANSACTION TIMING:
// -------------------------
//        Cycle 1     Cycle 2     Cycle 3     Cycle 4
//         _____       _____       _____       _____
//   clk__|     |_____|     |_____|     |_____|     |_____
//
// awvalid ▂▂▂▂▂▂▂▂▂▂▂▂▀▀▀▀▀▀▀▀▀▀▀▀▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂
// awready ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂
// awaddr  ----------------XXXX------------------------
//
// wvalid  ▂▂▂▂▂▂▂▂▂▂▂▂▀▀▀▀▀▀▀▀▀▀▀▀▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂
// wready  ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂
// wdata   ----------------DDDD------------------------
//
// bvalid  ▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
// bready  ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
// bresp   ----------------00--------------------------
//                     ▲
//                     │ csr_wen pulse (1 cycle)
//
// The write address (AW) and write data (W) channels may arrive in any order.
// This module latches whichever arrives first and waits for the other.
// Once both are received, it pulses csr_wen and asserts bvalid.
//
// READ TRANSACTION TIMING:
// ------------------------
//        Cycle 1     Cycle 2     Cycle 3
//         _____       _____       _____
//   clk__|     |_____|     |_____|     |_____
//
// arvalid ▂▂▂▂▂▂▂▂▂▂▂▂▀▀▀▀▀▀▀▀▀▀▀▀▂▂▂▂▂▂▂▂▂▂▂▂
// arready ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▂▂▂▂▂▂▂▂▂▂▂▂
// araddr  ----------------AAAA----------------
//                     ▲
//                     │ csr_ren pulse
//
// rvalid  ▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▀▀▀▀▀▀▀▀▀▀▀▀
// rready  ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
// rdata   ----------------DDDD----------------
// rresp   ----------------00------------------
//
// Read is simpler: on arvalid & arready, pulse csr_ren and capture csr_rdata.
// Assert rvalid the next cycle with the captured data.
//
// LATENCY:
// --------
// - Write: 2 cycles (AW+W handshake → bvalid)
// - Read: 2 cycles (AR handshake → rvalid)
//   Note: Read data is combinationally driven from csr_rdata, so CSR module
//   must provide registered output for timing closure.
//
// RESOURCE ESTIMATES:
// -------------------
//   - LUTs: ~100 (FSM and muxes)
//   - FFs: ~100 (latches and state)
//   - Critical path: csr_rdata → s_axi_rdata (combinational)
//
// INTEGRATION:
// ------------
// In Vivado block design:
//   - Connect s_axi_* ports to AXI Interconnect slave port
//   - AXI Interconnect master connects to M_AXI_GP0 of Zynq PS
//   - Set address range (e.g., 0x4000_0000 - 0x4000_0FFF for 4KB CSR space)
//
// =============================================================================

module axi_lite_slave #(
    // =========================================================================
    // PARAMETERS
    // =========================================================================
    // CSR_ADDR_WIDTH: Number of address bits exposed to CSR module
    // - 8 bits = 256 registers × 4 bytes = 1 KB address space
    // - Host address bits above CSR_ADDR_WIDTH are decoded externally
    //   (by AXI Interconnect address map)
    parameter CSR_ADDR_WIDTH = 8,
    
    // CSR_DATA_WIDTH: Register data width (always 32 for AXI4-Lite)
    parameter CSR_DATA_WIDTH = 32
)(
    // =========================================================================
    // CLOCK AND RESET
    // =========================================================================
    input  logic                        clk,      // AXI clock (e.g., 100 MHz)
    input  logic                        rst_n,    // Active-low async reset

    // =========================================================================
    // AXI4-LITE WRITE ADDRESS CHANNEL
    // =========================================================================
    // Carries the target register address for write operations.
    // awprot encodes privilege/security (typically ignored in simple designs).
    input  logic                        s_axi_awvalid,  // Address valid
    output logic                        s_axi_awready,  // This module can accept
    input  logic [CSR_ADDR_WIDTH-1:0]   s_axi_awaddr,   // Byte address (4B aligned)
    input  logic [2:0]                  s_axi_awprot,   // Protection (ignored)

    // =========================================================================
    // AXI4-LITE WRITE DATA CHANNEL
    // =========================================================================
    // Carries the data to be written. wstrb indicates which bytes are valid.
    // For 32-bit registers, wstrb is typically 4'b1111 (all bytes).
    input  logic                        s_axi_wvalid,   // Data valid
    output logic                        s_axi_wready,   // This module can accept
    input  logic [CSR_DATA_WIDTH-1:0]   s_axi_wdata,    // Write data (32 bits)
    input  logic [(CSR_DATA_WIDTH/8)-1:0] s_axi_wstrb,  // Byte enables (4 bits)

    // =========================================================================
    // AXI4-LITE WRITE RESPONSE CHANNEL
    // =========================================================================
    // Acknowledges write completion. bresp indicates success (OKAY) or error.
    output logic                        s_axi_bvalid,   // Response valid
    input  logic                        s_axi_bready,   // Master can accept
    output logic [1:0]                  s_axi_bresp,    // 2'b00=OKAY, 2'b10=SLVERR

    // =========================================================================
    // AXI4-LITE READ ADDRESS CHANNEL
    // =========================================================================
    // Carries the target register address for read operations.
    input  logic                        s_axi_arvalid,  // Address valid
    output logic                        s_axi_arready,  // This module can accept
    input  logic [CSR_ADDR_WIDTH-1:0]   s_axi_araddr,   // Byte address (4B aligned)
    input  logic [2:0]                  s_axi_arprot,   // Protection (ignored)

    // =========================================================================
    // AXI4-LITE READ DATA CHANNEL
    // =========================================================================
    // Returns read data with response status.
    output logic                        s_axi_rvalid,   // Data valid
    input  logic                        s_axi_rready,   // Master can accept
    output logic [CSR_DATA_WIDTH-1:0]   s_axi_rdata,    // Read data (32 bits)
    output logic [1:0]                  s_axi_rresp,    // 2'b00=OKAY

    // =========================================================================
    // CSR INTERFACE (to external CSR module)
    // =========================================================================
    // Simple register bus: single-cycle read/write operations.
    //
    // Write: csr_wen=1, csr_addr=address, csr_wdata=data (1 cycle)
    // Read:  csr_ren=1, csr_addr=address, csr_rdata valid same cycle
    //        (CSR module must have registered outputs or this becomes critical)
    output logic                        csr_wen,        // Write enable (1 cycle pulse)
    output logic [CSR_ADDR_WIDTH-1:0]   csr_addr,       // Register address
    output logic [CSR_DATA_WIDTH-1:0]   csr_wdata,      // Write data
    input  logic [CSR_DATA_WIDTH-1:0]   csr_rdata,      // Read data (combinational)
    output logic                        csr_ren,        // Read enable (1 cycle pulse)
    output logic                        axi_error       // Sticky error flag (unused)
);

    // =========================================================================
    // INTERNAL REGISTERS
    // =========================================================================
    // Latches to hold write address and data until both channels complete.
    // AXI4-Lite allows AW and W channels to arrive in any order.
    logic [CSR_ADDR_WIDTH-1:0] waddr_latch;   // Latched write address
    logic [CSR_DATA_WIDTH-1:0] wdata_latch;   // Latched write data
    logic                      got_waddr;      // AW channel handshake completed
    logic                      got_wdata;      // W channel handshake completed

    // =========================================================================
    // ERROR TRACKING
    // =========================================================================
    // Future: Set on SLVERR (e.g., invalid address). Currently unused.
    assign axi_error = 1'b0;

    // =========================================================================
    // WRITE CHANNEL FSM
    // =========================================================================
    // This is a simple state machine that:
    // 1. Accepts write address when awvalid && awready (latch address)
    // 2. Accepts write data when wvalid && wready (latch data)
    // 3. When BOTH are received, pulse csr_wen and assert bvalid
    // 4. Clear state when bvalid && bready (response acknowledged)
    //
    // Key insight: awready and wready are high when not holding pending data
    // and not currently responding. This allows both channels to handshake
    // in the same cycle or different cycles.
    
    // Ready signals: High when not holding pending data and no pending response
    assign s_axi_awready = !got_waddr && !s_axi_bvalid;
    assign s_axi_wready  = !got_wdata && !s_axi_bvalid;

    // Detect handshakes on each channel
    wire aw_handshake = s_axi_awvalid && s_axi_awready;  // Address accepted
    wire w_handshake  = s_axi_wvalid && s_axi_wready;    // Data accepted
    
    // Transaction complete when both channels have handshaken
    // (either this cycle or previously)
    wire both_complete = (got_waddr || aw_handshake) && (got_wdata || w_handshake);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset: Clear all latches and response
            got_waddr    <= 1'b0;
            got_wdata    <= 1'b0;
            waddr_latch  <= '0;
            wdata_latch  <= '0;
            s_axi_bvalid <= 1'b0;
            s_axi_bresp  <= 2'b00;  // OKAY
        end else begin
            // Response accepted: Clear state for next transaction
            if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
                got_waddr    <= 1'b0;
                got_wdata    <= 1'b0;
            end else begin
                // Capture write address when handshake occurs
                if (aw_handshake) begin
                    waddr_latch <= s_axi_awaddr;
                    got_waddr   <= 1'b1;
                end
                
                // Capture write data when handshake occurs
                if (w_handshake) begin
                    wdata_latch <= s_axi_wdata;
                    got_wdata   <= 1'b1;
                end
                
                // When both channels complete, issue CSR write and respond
                if (both_complete && !s_axi_bvalid) begin
                    s_axi_bvalid <= 1'b1;
                    s_axi_bresp  <= 2'b00;  // Always OKAY (no address decode errors)
                end
            end
        end
    end

    // =========================================================================
    // READ CHANNEL FSM
    // =========================================================================
    // Simpler than write: single channel (AR) triggers read.
    // 1. Accept address when arvalid && arready
    // 2. In same cycle, pulse csr_ren and capture csr_rdata
    // 3. Next cycle, assert rvalid with captured data
    // 4. Clear when rvalid && rready
    //
    // Note: This implementation captures csr_rdata synchronously, so there's
    // a 1-cycle latency from AR handshake to R response. The CSR module
    // should provide combinational read data for minimum latency.
    
    // Ready when not holding a pending response
    assign s_axi_arready = !s_axi_rvalid;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rdata  <= '0;
            s_axi_rresp  <= 2'b00;  // OKAY
        end else begin
            // Accept read request: Capture address and data, assert rvalid
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rresp  <= 2'b00;  // Always OKAY
                s_axi_rdata  <= csr_rdata;  // Combinational from CSR
            end
            
            // Response accepted: Clear for next transaction
            if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    // =========================================================================
    // CSR INTERFACE OUTPUTS
    // =========================================================================
    // Connect AXI transactions to the CSR module.
    //
    // Address muxing:
    //   - During read (arvalid): Use s_axi_araddr
    //   - During write (aw_handshake): Use s_axi_awaddr directly
    //   - Otherwise (latched write): Use waddr_latch
    //
    // This allows the CSR module to see the correct address for both
    // read and write operations with minimal latency.
    
    // Write enable: Pulse for one cycle when both AW and W complete
    assign csr_wen   = both_complete && !s_axi_bvalid;
    
    // Address: Prioritize read over write for concurrent access
    assign csr_addr  = s_axi_arvalid ? s_axi_araddr : 
                       (aw_handshake ? s_axi_awaddr : waddr_latch);
    
    // Write data: Use live data if available, otherwise latched
    assign csr_wdata = w_handshake ? s_axi_wdata : wdata_latch;
    
    // Read enable: Pulse for one cycle on AR handshake
    assign csr_ren   = s_axi_arvalid && s_axi_arready;

endmodule
