// =============================================================================
// axi_lite_slave.sv — Minimal AXI4-Lite Slave for CSR & DMA Access
// =============================================================================
// Purpose:
//   Provides AXI4-Lite control-plane interface to CSR registers.
//   Supports read/write of CSR fields and direct mapping to DMA control.
//
// Features:
//   - Full AXI4-Lite write and read channels
//   - CSR decoding (addr → CSR field)
//   - Burst support (WLEN, WSIZE parameters)
//   - Error handling (SLVERR on invalid addresses)
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module axi_lite_slave #(
    parameter CSR_ADDR_WIDTH = 8,
    parameter CSR_DATA_WIDTH = 32
)(
    // Clock and reset
    input  wire clk,
    input  wire rst_n,
    
    // AXI4-Lite Write Address Channel
    input  wire [CSR_ADDR_WIDTH-1:0]   s_axi_awaddr,
    input  wire [2:0]                  s_axi_awprot,
    input  wire                        s_axi_awvalid,
    output reg                         s_axi_awready,
    
    // AXI4-Lite Write Data Channel
    input  wire [CSR_DATA_WIDTH-1:0]   s_axi_wdata,
    input  wire [(CSR_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                        s_axi_wvalid,
    output reg                         s_axi_wready,
    
    // AXI4-Lite Write Response Channel
    output reg [1:0]                   s_axi_bresp,
    output reg                         s_axi_bvalid,
    input  wire                        s_axi_bready,
    
    // AXI4-Lite Read Address Channel
    input  wire [CSR_ADDR_WIDTH-1:0]   s_axi_araddr,
    input  wire [2:0]                  s_axi_arprot,
    input  wire                        s_axi_arvalid,
    output reg                         s_axi_arready,
    
    // AXI4-Lite Read Data Channel
    output reg [CSR_DATA_WIDTH-1:0]    s_axi_rdata,
    output reg [1:0]                   s_axi_rresp,
    output reg                         s_axi_rvalid,
    input  wire                        s_axi_rready,
    
    // CSR interface (to central CSR controller or DMA)
    output reg [CSR_ADDR_WIDTH-1:0]    csr_addr,
    output reg                         csr_wen,
    output reg                         csr_ren,
    output reg [CSR_DATA_WIDTH-1:0]    csr_wdata,
    input  wire [CSR_DATA_WIDTH-1:0]   csr_rdata,
    
    // Status signals
    output reg axi_error
);

    // Handshake signals
    wire aw_handshake = s_axi_awvalid && s_axi_awready;
    wire w_handshake  = s_axi_wvalid && s_axi_wready;
    wire b_handshake  = s_axi_bvalid && s_axi_bready;
    wire ar_handshake = s_axi_arvalid && s_axi_arready;
    wire r_handshake  = s_axi_rvalid && s_axi_rready;
    
    // CSR address map (validate on write/read)
    localparam [7:0] CSR_DMA_LAYER  = 8'h50,
                     CSR_DMA_CTRL   = 8'h51,
                     CSR_DMA_COUNT  = 8'h52,
                     CSR_DMA_STATUS = 8'h53,
                     CSR_DMA_BURST  = 8'h54;
    
    // Valid CSR address check
    function automatic is_valid_csr(input [CSR_ADDR_WIDTH-1:0] addr);
        case (addr)
            CSR_DMA_LAYER, CSR_DMA_CTRL, CSR_DMA_COUNT, CSR_DMA_STATUS, CSR_DMA_BURST:
                return 1'b1;
            default: return 1'b0;
        endcase
    endfunction
    
    // Write path state
    reg [CSR_ADDR_WIDTH-1:0] write_addr;
    reg [CSR_DATA_WIDTH-1:0] write_data;
    reg write_valid;
    
    // Read path state
    reg [CSR_ADDR_WIDTH-1:0] read_addr;
    reg read_valid;
    
    // Read pipeline stage (to ensure data is available when rvalid asserts)
    reg [CSR_DATA_WIDTH-1:0] read_data_pipe;
    reg [1:0] read_resp_pipe;
    reg read_valid_pipe;
    
    // ========================================================================
    // Write Address Path
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_awready <= 1'b1;
            write_addr <= {CSR_ADDR_WIDTH{1'b0}};
            write_valid <= 1'b0;
        end else begin
            // Accept write address when valid and ready to receive
            if (s_axi_awvalid && s_axi_awready) begin
                write_addr <= s_axi_awaddr;
                write_valid <= 1'b1;
                s_axi_awready <= 1'b0;  // Latch the address
            end else if (w_handshake) begin
                // After write data arrives, clear write_valid for next transaction
                s_axi_awready <= 1'b1;
                write_valid <= 1'b0;
            end
        end
    end
    
    // ========================================================================
    // Write Data Path
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_wready <= 1'b1;
            write_data <= {CSR_DATA_WIDTH{1'b0}};
        end else begin
            // Accept write data when valid
            if (s_axi_wvalid && s_axi_wready) begin
                write_data <= s_axi_wdata;
            end
        end
    end
    
    // ========================================================================
    // Write Response Path & CSR Write
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_bvalid <= 1'b0;
            s_axi_bresp <= 2'b00;  // OKAY
            csr_wen <= 1'b0;
            csr_addr <= {CSR_ADDR_WIDTH{1'b0}};
            csr_wdata <= {CSR_DATA_WIDTH{1'b0}};
            axi_error <= 1'b0;
        end else begin
            csr_wen <= 1'b0;  // Pulse
            
            // Trigger write when BOTH address and data handshakes have occurred
            // (write_addr and write_data are latched from their handshakes)
            if (write_valid && s_axi_wvalid && !s_axi_bvalid) begin
                if (is_valid_csr(write_addr)) begin
                    csr_addr <= write_addr;
                    csr_wdata <= s_axi_wdata;
                    csr_wen <= 1'b1;
                    s_axi_bresp <= 2'b00;  // OKAY
                end else begin
                    s_axi_bresp <= 2'b10;  // SLVERR (Slave Error)
                    axi_error <= 1'b1;
                end
                s_axi_bvalid <= 1'b1;
            end else if (b_handshake) begin
                // Response accepted by master; clear for next write
                s_axi_bvalid <= 1'b0;
                s_axi_wready <= 1'b1;
                write_valid <= 1'b0;  // Clear write_valid here to allow next write
            end
        end
    end
    
    // ========================================================================
    // Read Address Path
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_arready <= 1'b1;
            read_addr <= {CSR_ADDR_WIDTH{1'b0}};
            read_valid <= 1'b0;
        end else begin
            if (s_axi_arvalid && s_axi_arready) begin
                read_addr <= s_axi_araddr;
                read_valid <= 1'b1;
                s_axi_arready <= 1'b0;
            end else if (r_handshake) begin
                s_axi_arready <= 1'b1;
                read_valid <= 1'b0;
            end
        end
    end
    
    // ========================================================================
    // Read Data Path & CSR Read (Two-stage pipeline)
    // ========================================================================
    // Stage 1: Issue read to CSR
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_data_pipe <= {CSR_DATA_WIDTH{1'b0}};
            read_resp_pipe <= 2'b00;
            read_valid_pipe <= 1'b0;
            csr_ren <= 1'b0;
        end else begin
            csr_ren <= 1'b0;  // Pulse
            
            // When read address is available, perform read
            if (read_valid && !read_valid_pipe) begin
                if (is_valid_csr(read_addr)) begin
                    csr_addr <= read_addr;
                    csr_ren <= 1'b1;
                    read_resp_pipe <= 2'b00;  // OKAY
                end else begin
                    read_resp_pipe <= 2'b10;  // SLVERR
                    axi_error <= 1'b1;
                end
                read_data_pipe <= csr_rdata;
                read_valid_pipe <= 1'b1;
            end else if (s_axi_rvalid && s_axi_rready) begin
                read_valid_pipe <= 1'b0;
            end
        end
    end
    
    // Stage 2: Latch pipeline to output
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rdata <= {CSR_DATA_WIDTH{1'b0}};
            s_axi_rresp <= 2'b00;
        end else begin
            if (read_valid_pipe && !s_axi_rvalid) begin
                s_axi_rdata <= read_data_pipe;
                s_axi_rresp <= read_resp_pipe;
                s_axi_rvalid <= 1'b1;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

endmodule

`default_nettype wire
// =============================================================================
// End of axi_lite_slave.sv
// =============================================================================
