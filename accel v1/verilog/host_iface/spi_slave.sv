// =============================================================================
// spi_slave.sv — Minimal SPI Slave for DMA Control (Optional Phase 4b)
// =============================================================================
// Purpose:
//   Provides a simpler alternative to AXI4-Lite for embedded hosts with SPI-only interfaces.
//   Maps SPI transactions to CSR reads/writes and optional DMA bursts.
//
// Features:
//   - SPI Mode 0 (CPOL=0, CPHA=0)
//   - Clock/CS management
//   - Command/address/data sequencing
//   - DMA FIFO muxing
//
// Protocol:
//   TX: [CMD(8)] [ADDR(16)] [DATA(32)]  (for writes)
//   RX: [DATA(32)]  (for reads/responses)
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module spi_slave #(
    parameter CMD_WIDTH = 8,
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 32
)(
    // SPI Interface
    input  wire spi_clk,
    input  wire spi_cs_n,
    input  wire spi_mosi,
    output reg  spi_miso,
    
    // System Interface
    input  wire clk,
    input  wire rst_n,
    
    // CSR Interface (to accel_top CSR bus)
    output reg [7:0]  csr_addr,
    output reg [31:0] csr_wdata,
    output reg        csr_wen,
    output reg        csr_ren,
    input  wire [31:0] csr_rdata,
    
    // DMA FIFO Interface (optional, for burst writes)
    output reg [31:0] dma_fifo_wdata,
    output reg        dma_fifo_wen,
    input  wire       dma_fifo_full,
    
    // Status
    output reg [31:0] spi_bytes_received,
    output reg [31:0] spi_errors
);

    // ========================================================================
    // SPI Clock Domain Synchronization
    // ========================================================================
    
    reg [2:0] spi_clk_r;
    reg [2:0] cs_r;
    wire spi_clk_edge = (spi_clk_r[1:0] == 2'b01);  // Rising edge
    wire cs_falling_edge = (cs_r[1:0] == 2'b10);    // Falling edge (start)
    wire cs_rising_edge = (cs_r[1:0] == 2'b01);     // Rising edge (end)
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spi_clk_r <= 3'd0;
            cs_r <= 3'd0;
        end else begin
            spi_clk_r <= {spi_clk_r[1:0], spi_clk};
            cs_r <= {cs_r[1:0], spi_cs_n};
        end
    end
    
    // ========================================================================
    // SPI Shift Register & Command State Machine
    // ========================================================================
    
    reg [63:0] spi_rx_shift;  // 64-bit shift: [CMD(8)][ADDR(16)][DATA(32)][PAD(8)]
    reg [63:0] spi_tx_shift;
    reg [5:0]  spi_bit_count;
    
    localparam [2:0] SPI_IDLE    = 3'd0,
                     SPI_CMD     = 3'd1,
                     SPI_ADDR    = 3'd2,
                     SPI_DATA    = 3'd3,
                     SPI_EXECUTE = 3'd4;
    
    reg [2:0] spi_state;
    reg [7:0] cmd_byte;
    reg [15:0] addr_word;
    reg [31:0] data_word;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spi_state <= SPI_IDLE;
            spi_bit_count <= 6'd0;
            spi_rx_shift <= 64'd0;
            spi_tx_shift <= 64'd0;
            cmd_byte <= 8'd0;
            addr_word <= 16'd0;
            data_word <= 32'd0;
            csr_wen <= 1'b0;
            csr_ren <= 1'b0;
            dma_fifo_wen <= 1'b0;
            spi_bytes_received <= 32'd0;
        end else begin
            // Default: disable writes
            csr_wen <= 1'b0;
            csr_ren <= 1'b0;
            dma_fifo_wen <= 1'b0;
            spi_miso <= spi_tx_shift[63];  // MSB first
            
            // CS falling edge: start transaction
            if (cs_falling_edge) begin
                spi_state <= SPI_CMD;
                spi_bit_count <= 6'd0;
                spi_rx_shift <= 64'd0;
            end
            
            // SPI clock edge: shift in/out
            if (spi_clk_edge && !spi_cs_n) begin
                spi_rx_shift <= {spi_rx_shift[62:0], spi_mosi};
                spi_tx_shift <= {spi_tx_shift[62:0], 1'b0};
                spi_bit_count <= spi_bit_count + 1'b1;
                
                case (spi_state)
                    SPI_CMD: begin
                        if (spi_bit_count == 6'd7) begin
                            cmd_byte <= {spi_rx_shift[62:0], spi_mosi};
                            spi_state <= SPI_ADDR;
                            spi_bit_count <= 6'd0;
                            spi_tx_shift <= 64'd0;
                        end
                    end
                    
                    SPI_ADDR: begin
                        if (spi_bit_count == 6'd15) begin
                            addr_word <= {spi_rx_shift[62:48], spi_mosi};
                            spi_state <= SPI_DATA;
                            spi_bit_count <= 6'd0;
                        end
                    end
                    
                    SPI_DATA: begin
                        if (spi_bit_count == 6'd31) begin
                            data_word <= {spi_rx_shift[62:32], spi_mosi};
                            spi_state <= SPI_EXECUTE;
                            spi_bit_count <= 6'd0;
                        end
                    end
                    
                    default: begin
                        if (spi_bit_count == 6'd31) begin
                            // Dummy read
                            spi_state <= SPI_EXECUTE;
                            spi_bit_count <= 6'd0;
                        end
                    end
                endcase
            end
            
            // CS rising edge: end transaction, execute command
            if (cs_rising_edge) begin
                spi_bytes_received <= spi_bytes_received + 1;
                
                case (cmd_byte[7:4])
                    4'h0: begin  // CSR Write
                        csr_addr <= cmd_byte[7:0];
                        csr_wdata <= data_word;
                        csr_wen <= 1'b1;
                    end
                    
                    4'h1: begin  // CSR Read
                        csr_addr <= cmd_byte[7:0];
                        csr_ren <= 1'b1;
                        spi_tx_shift <= {csr_rdata, 32'd0};
                    end
                    
                    4'h2: begin  // DMA Burst Write
                        if (!dma_fifo_full) begin
                            dma_fifo_wdata <= data_word;
                            dma_fifo_wen <= 1'b1;
                        end else begin
                            spi_errors <= spi_errors + 1;
                        end
                    end
                    
                    default: begin
                        spi_errors <= spi_errors + 1;
                    end
                endcase
                
                spi_state <= SPI_IDLE;
            end
        end
    end

endmodule

`default_nettype wire
// =============================================================================
// End of spi_slave.sv
// =============================================================================
