// =============================================================================
// axi_dma_bridge.sv — AXI Write-Burst to DMA FIFO Bridge
// =============================================================================
// Purpose:
//   Maps AXI4-Full write bursts to the bsr_dma block-write FIFO.
//   Handles multi-beat bursts and assembles 32-bit words for DMA.
//
// Features:
//   - AXI4 write address and data channels
//   - Burst length support (WLEN up to 256)
//   - Write strobe (WSTRB) handling
//   - Flow control and error reporting
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module axi_dma_bridge #(
    parameter DATA_WIDTH = 32,
    parameter ADDR_WIDTH = 32,
    parameter DMA_FIFO_DEPTH = 64,
    parameter DMA_FIFO_PTR_W = 6
)(
    // Clock and reset
    input  wire clk,
    input  wire rst_n,
    
    // AXI4 Write Address Channel
    input  wire [ADDR_WIDTH-1:0]   s_axi_awaddr,
    input  wire [1:0]              s_axi_awburst,
    input  wire [7:0]              s_axi_awlen,
    input  wire [2:0]              s_axi_awsize,
    input  wire                    s_axi_awvalid,
    output reg                     s_axi_awready,
    
    // AXI4 Write Data Channel
    input  wire [DATA_WIDTH-1:0]   s_axi_wdata,
    input  wire [(DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  wire                    s_axi_wlast,
    input  wire                    s_axi_wvalid,
    output reg                     s_axi_wready,
    
    // AXI4 Write Response Channel
    output reg [1:0]               s_axi_bresp,
    output reg                     s_axi_bvalid,
    input  wire                    s_axi_bready,
    
    // DMA FIFO Interface (32-bit words, LSB-first)
    output reg [31:0]              dma_fifo_wdata,
    output reg                     dma_fifo_wen,
    input  wire                    dma_fifo_full,
    input  wire [DMA_FIFO_PTR_W:0] dma_fifo_count,
    
    // Status & error
    output reg                     axi_error,
    output reg [31:0]              words_written
);

    // State machine for burst handling
    localparam [1:0] IDLE       = 2'd0,
                     BURST_DATA = 2'd1,
                     WAIT_RESP  = 2'd2;
    
    reg [1:0] state;
    reg [7:0] beat_count;
    reg [7:0] burst_len;
    reg [1:0] burst_type;
    
    // ========================================================================
    // Write Address Latch
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_awready <= 1'b1;
            burst_len <= 8'd0;
            burst_type <= 2'd0;
        end else begin
            case (state)
                IDLE: begin
                    if (s_axi_awvalid) begin
                        burst_len <= s_axi_awlen;
                        burst_type <= s_axi_awburst;
                        s_axi_awready <= 1'b0;
                        // Transition to receive data
                    end
                end
                default: s_axi_awready <= 1'b0;
            endcase
        end
    end
    
    // ========================================================================
    // Write Data Path & DMA FIFO Enqueue
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            s_axi_wready <= 1'b0;
            s_axi_bvalid <= 1'b0;
            dma_fifo_wen <= 1'b0;
            dma_fifo_wdata <= 32'd0;
            beat_count <= 8'd0;
            axi_error <= 1'b0;
            words_written <= 32'd0;
        end else begin
            dma_fifo_wen <= 1'b0;  // Pulse
            
            case (state)
                IDLE: begin
                    s_axi_wready <= 1'b1;
                    if (s_axi_awvalid && s_axi_awready) begin
                        state <= BURST_DATA;
                        beat_count <= 8'd0;
                    end
                end
                
                BURST_DATA: begin
                    // Accept write data and enqueue to DMA FIFO
                    if (s_axi_wvalid && s_axi_wready) begin
                        if (!dma_fifo_full) begin
                            // Enqueue 32-bit word to DMA FIFO
                            dma_fifo_wdata <= s_axi_wdata;
                            dma_fifo_wen <= 1'b1;
                            words_written <= words_written + 1;
                            beat_count <= beat_count + 1;
                            
                            // Check if burst complete
                            if (s_axi_wlast || (beat_count == burst_len)) begin
                                state <= WAIT_RESP;
                                s_axi_wready <= 1'b0;
                            end
                        end else begin
                            // DMA FIFO full: stall
                            s_axi_wready <= 1'b0;
                            axi_error <= 1'b1;
                        end
                    end else begin
                        s_axi_wready <= 1'b1;
                    end
                end
                
                WAIT_RESP: begin
                    // Send write response
                    s_axi_bvalid <= 1'b1;
                    s_axi_bresp <= axi_error ? 2'b11 : 2'b00;  // SLVERR or OKAY
                    if (s_axi_bready) begin
                        s_axi_bvalid <= 1'b0;
                        state <= IDLE;
                        s_axi_awready <= 1'b1;
                        axi_error <= 1'b0;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
// =============================================================================
// End of axi_dma_bridge.sv
// =============================================================================
