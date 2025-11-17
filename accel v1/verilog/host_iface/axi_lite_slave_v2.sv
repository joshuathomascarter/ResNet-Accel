// =============================================================================
// axi_lite_slave.sv — Minimal AXI4-Lite Slave for CSR & DMA Access
// =============================================================================
// Simplified version with clearer state management
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
    input  wire [2:0]                  s_axi_awprot, // MAKE MORE COMPLEX USE THIS PROTECTION
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
    
    // CSR interface
    output reg [CSR_ADDR_WIDTH-1:0]    csr_addr,
    output reg                         csr_wen,
    output reg                         csr_ren,
    output reg [CSR_DATA_WIDTH-1:0]    csr_wdata,
    input  wire [CSR_DATA_WIDTH-1:0]   csr_rdata,
    
    // Status signals
    output reg axi_error
);

    // CSR address map
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
    
    // ========================================================================
    // Write path state
    // ========================================================================
    reg [CSR_ADDR_WIDTH-1:0] w_addr;
    reg [CSR_DATA_WIDTH-1:0] w_data;
    reg w_addr_valid, w_data_valid;  // Flags to track if addr/data have arrived
    reg [2:0] w_prot;  // Protection bits
    
    // ========================================================================
    // Write Address Path
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_awready <= 1'b1;
            w_addr <= {CSR_ADDR_WIDTH{1'b0}};
            w_addr_valid <= 1'b0;
        end else begin
            // Accept write address when master is offering it
            if (s_axi_awvalid && s_axi_awready) begin
                w_addr <= s_axi_awaddr;
                w_prot <= s_axi_awprot;  
                w_addr_valid <= 1'b1;
                s_axi_awready <= 1'b0;  // Block new addresses until after response
            end
        end
    end
    
    // ========================================================================
    // Write Data Path
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_wready <= 1'b1;
            w_data <= {CSR_DATA_WIDTH{1'b0}};
            w_data_valid <= 1'b0;
        end else begin
            // Accept write data when master is offering it
            if (s_axi_wvalid && s_axi_wready) begin
                w_data <= s_axi_wdata;
                w_data_valid <= 1'b1;
                s_axi_wready <= 1'b0;  // Block new data until after response
            end
        end
    end
    
    // ========================================================================
    // Write Response Path & CSR Write
    // ========================================================================
    reg write_in_progress;  // Track if we fired a write
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_bvalid <= 1'b0;
            s_axi_bresp <= 2'b00;
            csr_wen <= 1'b0;
            csr_addr <= {CSR_ADDR_WIDTH{1'b0}};
            csr_wdata <= {CSR_DATA_WIDTH{1'b0}};
            axi_error <= 1'b0;
            write_in_progress <= 1'b0;
        end else begin
            csr_wen <= 1'b0;  // One-cycle pulse
            
            // When both address and data are valid, perform the write
            if (w_addr_valid && w_data_valid && !s_axi_bvalid && !write_in_progress) begin
                
                // Simple check: Only privileged can write DMA_CTRL
                if (w_addr == CSR_DMA_CTRL && !w_prot[0]) begin
                    // Unprivileged trying to write DMA_CTRL → DENY
                    s_axi_bresp <= 2'b10;  // SLVERR
                    axi_error <= 1'b1;
                    s_axi_bvalid <= 1'b1;
                    write_in_progress <= 1'b1;
                end else begin
                    // Normal case statement (existing code)
                    case (w_addr)
                        CSR_DMA_LAYER, CSR_DMA_CTRL, CSR_DMA_COUNT, CSR_DMA_STATUS, CSR_DMA_BURST: begin
                            csr_addr <= w_addr;
                            csr_wdata <= w_data;
                            csr_wen <= 1'b1;
                            s_axi_bresp <= 2'b00;  // OKAY
                            axi_error <= 1'b0;
                        end
                        default: begin
                            s_axi_bresp <= 2'b11;  // SLVERR
                            axi_error <= 1'b1;
                        end
                    endcase
                    s_axi_bvalid <= 1'b1;
                    write_in_progress <= 1'b1;  // Prevent duplicate writes
                end
            end
            
            // After response is sent and accepted, clear for next write
            if (s_axi_bvalid && s_axi_bready) begin
                write_in_progress <= 1'b0;
            end
        end
    end
    
    // Clear write-pending flags after response is accepted
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // nothing
        end else begin
            if (s_axi_bvalid && s_axi_bready) begin
                s_axi_awready <= 1'b1;
                s_axi_wready <= 1'b1;
                s_axi_bvalid <= 1'b0;  // CLEAR bvalid here!
                w_addr_valid <= 1'b0;  // Clear for next write
                w_data_valid <= 1'b0;
            end
        end
    end
    
    // ========================================================================
    // Read path state
    // ========================================================================
    reg [CSR_ADDR_WIDTH-1:0] r_addr;
    reg r_addr_valid;
    
    // ========================================================================
    // Read Address Path
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_arready <= 1'b1;
            r_addr <= {CSR_ADDR_WIDTH{1'b0}};
            r_addr_valid <= 1'b0;
        end else begin
            // Accept read address when master is offering it
            if (s_axi_arvalid && s_axi_arready) begin
                r_addr <= s_axi_araddr;
                r_addr_valid <= 1'b1;
                s_axi_arready <= 1'b0;  // Block new addresses until data sent
            end else if (s_axi_rvalid && s_axi_rready) begin
                // After data accepted, prepare for next read
                s_axi_arready <= 1'b1;
                r_addr_valid <= 1'b0;
            end
        end
    end
    
    // ========================================================================
    // Read Data Path & CSR Read
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rdata <= {CSR_DATA_WIDTH{1'b0}};
            s_axi_rresp <= 2'b00;
            csr_ren <= 1'b0;
        end else begin
            csr_ren <= 1'b0;  // One-cycle pulse
            
            // When read address is valid and we're not already sending data
            if (r_addr_valid && !s_axi_rvalid) begin
                case (r_addr)
                    CSR_DMA_LAYER, CSR_DMA_CTRL, CSR_DMA_COUNT, CSR_DMA_STATUS, CSR_DMA_BURST: begin
                        csr_addr <= r_addr;
                        csr_ren <= 1'b1;
                        s_axi_rresp <= 2'b00;  // OKAY
                        s_axi_rdata <= csr_rdata;  // Combinational read
                    end
                    default: begin
                        s_axi_rresp <= 2'b11;  // SLVERR
                        s_axi_rdata <= {CSR_DATA_WIDTH{1'b0}};
                    end
                endcase
                s_axi_rvalid <= 1'b1;
            end
            // Data is held until master accepts (rready)
        end
    end

endmodule

`default_nettype wire
