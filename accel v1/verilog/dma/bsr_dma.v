// =============================================================================
// bsr_dma.v — BSR (Block Sparse Row) DMA Engine for Sparse Accelerator
// =============================================================================
// Purpose:
//   Transfers BSR-format sparse matrix data from host via UART into on-chip BRAMs:
//   - row_ptr[]: cumulative block counts (sparse metadata)
//   - col_idx[]: column position of each block
//   - blocks[]:  8×8 INT8 weight blocks
//
// Features:
//   - UART-based packet protocol for host transfers
//   - Multi-layer support: independently load different network layers
//   - Configurable BRAM write interface
//   - CRC-32 protection (optional)
//   - Status reporting via CSR interface
//
// Architecture:
//   ┌─────────────┐
//   │  UART RX    │  (receives packets from host)
//   └──────┬──────┘
//          │
//   ┌──────▼──────────┐
//   │  Packet Parser  │  (interprets DMA commands)
//   └──────┬──────────┘
//          │
//   ┌──────▼──────────────┐
//   │  BRAM Write Arbiter │  (addresses row_ptr/col_idx/blocks)
//   └──────┬───────────────┘
//          │
//   ┌──────▼──────────────┐
//   │  BRAM Interfaces   │  (writes to on-chip memories)
//   └────────────────────┘
//
// Packet Format (CSR-driven):
//   [CSR_ADDR] = 0x50: Layer selection (0-7)
//   [CSR_ADDR] = 0x51: DMA control (start, reset, etc.)
//   [CSR_ADDR] = 0x52: Write count (blocks loaded)
//   [CSR_ADDR] = 0x53: Status flags
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module bsr_dma #(
    parameter DATA_WIDTH        = 8,           // INT8 weights
    parameter ADDR_WIDTH        = 16,          // BRAM address width
    parameter WORD_BYTES        = 4,           // bytes per word for BRAM width
    parameter MAX_LAYERS        = 8,           // Support 0-7 network layers
    parameter MAX_BLOCKS        = 65536,       // 2^16 blocks per layer
    parameter BLOCK_SIZE        = 64,          // 8×8 blocks = 64 bytes
    parameter ROW_PTR_DEPTH     = 256,         // For 2048 output features / 8
    parameter COL_IDX_DEPTH     = 65536,       // One per block
    parameter BLOCK_DEPTH       = 65536 * 64,  // Total block data (in bytes)
    parameter BANKS             = 1,
    parameter ENABLE_CRC        = 0             // CRC32 protection (0=disabled for now)
)(
    // Clock and reset
    input  wire clk,
    input  wire rst_n,
    
    // UART RX interface (from uart_rx module)
    input  wire [7:0]  uart_rx_data,
    input  wire        uart_rx_valid,
    output wire        uart_rx_ready,
    
    // UART TX interface (for status responses)
    output reg [7:0]   uart_tx_data,
    output reg         uart_tx_valid,
    input  wire        uart_tx_ready,
    
    // CSR interface (for DMA control/status)
    input  wire [7:0]  csr_addr,
    input  wire        csr_wen,
    input  wire [31:0] csr_wdata,
    output reg [31:0]  csr_rdata,
    
    // row_ptr BRAM write interface
    output reg         row_ptr_we,
    output reg [ADDR_WIDTH-1:0] row_ptr_waddr,
    output reg [31:0]  row_ptr_wdata,
    
    // col_idx BRAM write interface
    output reg         col_idx_we,
    output reg [ADDR_WIDTH-1:0]  col_idx_waddr,
    output reg [15:0]  col_idx_wdata,
    
    // blocks BRAM write interface
    // 32-bit word-based block write bus
    output reg         block_we,
    output reg [ADDR_WIDTH+4:0] block_waddr,  // word address (divide by 4)
    output reg [31:0]  block_wdata,
    
    // Status outputs
    output reg         dma_busy,
    output reg         dma_done,
    output reg         dma_error,
    output reg [31:0]  blocks_written
);

    // ========================================================================
    // CSR Register Address Map
    // ========================================================================
    // Move DMA CSR addresses into unused CSR region to avoid collisions
    localparam [7:0] CSR_DMA_LAYER  = 8'h50,  // Layer selection
                     CSR_DMA_CTRL   = 8'h51,  // Control (start, reset)
                     CSR_DMA_COUNT  = 8'h52,  // Blocks written
                     CSR_DMA_STATUS = 8'h53;  // Status flags
    localparam [7:0] CSR_DMA_BURST  = 8'h54;  // Burst configuration (enable + length)

    localparam integer CSR_DMA_CTRL_WORD_MODE_BIT = 2; // bit 2 enables UART 32-bit word mode
    
    // ========================================================================
    // Packet Protocol State Machine
    // ========================================================================
    typedef enum logic [3:0] {
        IDLE            = 4'd0,   // Waiting for packet start
        RX_LAYER_CMD    = 4'd1,   // Receiving layer switch command
        RX_HDR_INIT     = 4'd2,   // Initialize header reception
        RX_HDR_DATA     = 4'd3,   // Receive header fields (three 32-bit words)
        RX_ROW_PTR_INIT = 4'd4,   // Initialize row_ptr load
        RX_ROW_PTR_DATA = 4'd5,   // Receive row_ptr entries (32-bit)
        RX_COL_IDX_INIT = 4'd6,   // Initialize col_idx load
        RX_COL_IDX_DATA = 4'd7,   // Receive col_idx entries (16-bit)
        RX_BLOCK_INIT   = 4'd8,   // Initialize block data load
        RX_BLOCK_DATA   = 4'd9,   // Receive 64-byte blocks (8×8 INT8)
        RX_CRC_CHK      = 4'd10,  // Verify CRC32 if enabled
        TX_STATUS       = 4'd11,  // Send status response
        DONE            = 4'd12   // Complete
    } state_t;
    
    state_t state, next_state;
    
    // ========================================================================
    // Internal Registers
    // ========================================================================
    reg [2:0]   active_layer;           // Currently loading layer (0-7)
    reg [31:0]  blocks_in_layer;        // Blocks loaded for this layer
    
    // Packet reception
    // Accumulators for multi-byte fields (byte-assembled into words)
    reg [31:0]  rx_word_buf;            // packs 4 bytes (LSB at bit0)
    reg [15:0]  rx_halfword;            // packs 2 bytes for 16-bit fields
    reg [1:0]   rx_byte_count;          // Count for bytes assembled (0..3)
    reg [31:0]  rx_payload;             // Assembled multi-byte value (word/halfword alias)
    // temp for assembled values
    reg [31:0]  assembled_word;
    
    // Block tracking
    reg [31:0]  block_byte_count;       // 0-63 for per-block reception
    reg [31:0]  current_block_idx;      // Global block index
    reg [ADDR_WIDTH-1:0] row_ptr_addr;  // Current write address for row_ptr
    reg [ADDR_WIDTH-1:0] col_idx_addr;  // Current write address for col_idx
    reg [ADDR_WIDTH+6:0] block_addr;    // Current write address for blocks
    // word-addressed block memory (words of 4 bytes)
    reg [ADDR_WIDTH+4:0] block_addr_word;    // word address (originally +6:0 for bytes)
    
    // CRC accumulator
    reg [31:0]  crc_accumulator;
    // UART 32-bit word mode, enables internal packing behavior
    reg         uart_word_mode_en;
    // Burst control registers
    reg burst_enable;
    reg [7:0] burst_len;

        // Block write FIFO for assembled 32-bit words (de-couples UART byte arrival from BRAM writes)
        localparam BLOCK_WRITE_FIFO_DEPTH = 64;
        localparam BLOCK_WRITE_FIFO_PTR_W = $clog2(BLOCK_WRITE_FIFO_DEPTH);
        reg [31:0] block_write_fifo_mem [0:BLOCK_WRITE_FIFO_DEPTH-1];
        reg [BLOCK_WRITE_FIFO_PTR_W-1:0] block_write_fifo_wr_ptr, block_write_fifo_rd_ptr;
        reg [BLOCK_WRITE_FIFO_PTR_W:0] block_write_fifo_count;
        wire block_write_fifo_full = (block_write_fifo_count == BLOCK_WRITE_FIFO_DEPTH);
        wire block_write_fifo_empty = (block_write_fifo_count == 0);
        reg [31:0] block_write_fifo_q;
        reg block_write_fifo_q_valid;

    // CRC function (IEEE CRC-32 polynomial 0x04C11DB7) — byte update
    function automatic [31:0] crc32_byte(input [31:0] cur, input [7:0] data);
        reg [31:0] c;
        reg [7:0] d;
        integer i;
        begin
            c = cur;
            d = data;
            for (i = 0; i < 8; i = i + 1) begin
                if ((c[31] ^ d[0]) == 1'b1) begin
                    c = (c << 1) ^ 32'h04C11DB7;
                end else begin
                    c = (c << 1);
                end
                d = d >> 1;
            end
            crc32_byte = c;
        end
    endfunction

    // Tiny input FIFO for UART RX flow control
    localparam FIFO_DEPTH = 16;
    localparam FIFO_PTR_W = $clog2(FIFO_DEPTH); // compute pointer width
    reg [7:0] fifo_mem [0:FIFO_DEPTH-1];
    reg [FIFO_PTR_W-1:0] fifo_wr_ptr, fifo_rd_ptr;
    reg [FIFO_PTR_W:0] fifo_count; // counts 0..FIFO_DEPTH
    wire fifo_full = (fifo_count == FIFO_DEPTH);
    wire fifo_empty = (fifo_count == 0);
    // Registered FIFO output for synchronous read semantics
    reg [7:0] fifo_q;
    reg fifo_get;      // request a synchronous read from FIFO into fifo_q
    reg fifo_consume;  // indicate we've consumed the registered fifo_q
    reg fifo_q_valid;   // indicates fifo_q contains a valid byte

    // Header fields
    reg [31:0] hdr_num_block_rows;
    reg [31:0] hdr_num_block_cols;
    reg [31:0] hdr_total_blocks;
    reg [1:0] hdr_word_idx; // 0..2
    reg [31:0] row_ptr_remaining;
    reg [31:0] col_idx_remaining;
    reg [31:0] block_remaining;
    // Assembled host CRC (LSB-first assembly) for verification
    reg [31:0] host_crc;
    
    // ========================================================================
    // State Machine Sequential Logic
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            active_layer <= 3'd0;
            blocks_in_layer <= 32'd0;
            blocks_written <= 32'd0;
            dma_busy <= 1'b0;
            dma_done <= 1'b0;
            dma_error <= 1'b0;
            rx_byte_count <= 2'd0;
            rx_word_buf <= 32'd0;
            rx_halfword <= 16'd0;
            fifo_wr_ptr <= 0;
            fifo_rd_ptr <= 0;
            fifo_count <= 0;
            fifo_get <= 1'b0;
            fifo_consume <= 1'b0;
            fifo_q_valid <= 1'b0;
            hdr_num_block_rows <= 32'd0;
            hdr_num_block_cols <= 32'd0;
            hdr_total_blocks <= 32'd0;
            hdr_word_idx <= 2'd0;
            block_byte_count <= 32'd0;
            current_block_idx <= 32'd0;
            row_ptr_addr <= {ADDR_WIDTH{1'b0}};
            col_idx_addr <= {ADDR_WIDTH{1'b0}};
            block_addr <= {ADDR_WIDTH+7{1'b0}};
            block_addr_word <= {ADDR_WIDTH+4{1'b0}};
            crc_accumulator <= 32'hFFFF_FFFF;  // CRC32 initial value
            // Block write FIFO reset
            block_write_fifo_wr_ptr <= {BLOCK_WRITE_FIFO_PTR_W{1'b0}};
            block_write_fifo_rd_ptr <= {BLOCK_WRITE_FIFO_PTR_W{1'b0}};
            block_write_fifo_count <= {BLOCK_WRITE_FIFO_PTR_W+1{1'b0}};
            block_write_fifo_q <= 32'd0;
            block_write_fifo_q_valid <= 1'b0;
            host_crc <= 32'd0;
            // Clear burst control
            burst_enable <= 1'b0;
            burst_len <= 8'd0;
        end else begin
            state <= next_state;
            // CSR_BURST write handling at runtime
            if (csr_wen && csr_addr == CSR_DMA_BURST) begin
                burst_enable <= csr_wdata[0];
                burst_len <= csr_wdata[7:0];
            end
            
                    // Default: disable writes
            row_ptr_we <= 1'b0;
            col_idx_we <= 1'b0;
            block_we <= 1'b0;
            uart_tx_valid <= 1'b0;
                    // Default: avoid FIFO operations unless handled in state
                    fifo_get <= 1'b0;
                    fifo_consume <= 1'b0;
                    // Dequeue assembled 32-bit block write FIFO if available
                    if (!block_write_fifo_q_valid && (block_write_fifo_count > 0)) begin
                        block_write_fifo_q <= block_write_fifo_mem[block_write_fifo_rd_ptr];
                        block_write_fifo_rd_ptr <= block_write_fifo_rd_ptr + 1;
                        block_write_fifo_q_valid <= 1'b1;
                        block_write_fifo_count <= block_write_fifo_count - 1;
                    end
                    // If a word-ready from FIFO is available and we are receiving block data, write it
                    if (block_write_fifo_q_valid && (state == RX_BLOCK_DATA)) begin
                        block_we <= 1'b1;
                        block_wdata <= block_write_fifo_q;
                        block_waddr <= block_addr_word;
                        block_addr_word <= block_addr_word + 1;
                        block_write_fifo_q_valid <= 1'b0;
                    end
            
            case (state)
                IDLE: begin
                    if (csr_wen && csr_addr == CSR_DMA_CTRL && csr_wdata[0]) begin
                        dma_busy <= 1'b1;
                        dma_done <= 1'b0;
                        dma_error <= 1'b0;
                        blocks_in_layer <= 32'd0;
                        block_byte_count <= 32'd0;
                        current_block_idx <= 32'd0;
                        // Latch word mode setting at start; keeps current behavior by default
                        uart_word_mode_en <= csr_wdata[CSR_DMA_CTRL_WORD_MODE_BIT];
                        // Reset header and counters
                        hdr_num_block_rows <= 32'd0;
                        hdr_num_block_cols <= 32'd0;
                        hdr_total_blocks <= 32'd0;
                        hdr_word_idx <= 2'd0;
                        // Reset CRC accumulator at start of transfer
                        crc_accumulator <= 32'hFFFF_FFFF;
                        // Reset assembled host CRC
                        host_crc <= 32'd0;
                        // Reset burst config
                        burst_enable <= 1'b0;
                        burst_len <= 8'd0;
                    end
                end
                
                RX_LAYER_CMD: begin
                    // If we already have a registered FIFO byte, consume it
                    if (fifo_q_valid) begin
                        active_layer <= fifo_q[2:0];
                        fifo_consume <= 1'b1; // release registered byte
                        rx_byte_count <= 2'd0;
                    end else if (fifo_count > 0) begin
                        // Request a synchronous read from FIFO into fifo_q
                        fifo_get <= 1'b1;
                    end else begin
                        fifo_get <= 1'b0;
                        fifo_consume <= 1'b0;
                    end
                end

                RX_HDR_INIT: begin
                    // Prepare to receive 3 header words
                    hdr_word_idx <= 2'd0;
                    rx_byte_count <= 2'd0;
                    rx_word_buf <= 32'd0;
                end

                RX_HDR_DATA: begin
                    // Wait for registered FIFO byte to become available and assemble 32-bit words LSB-first
                    if (fifo_q_valid) begin
                        // write byte into word buffer
                        rx_word_buf[rx_byte_count * 8 +: 8] <= fifo_q;
                        // consume the registered byte
                        fifo_consume <= 1'b1;
                        if (rx_byte_count == 2'd3) begin
                            // header word complete
                            case (hdr_word_idx)
                                2'd0: hdr_num_block_rows <= {fifo_q, rx_word_buf[23:0]};
                                2'd1: hdr_num_block_cols <= {fifo_q, rx_word_buf[23:0]};
                                2'd2: hdr_total_blocks <= {fifo_q, rx_word_buf[23:0]};
                            endcase
                            hdr_word_idx <= hdr_word_idx + 1;
                            rx_byte_count <= 2'd0;
                            rx_word_buf <= 32'd0;
                            // clear any pending get/consume if present
                            fifo_get <= 1'b0;
                            fifo_consume <= 1'b0;
                            // NOTE: Do NOT assign remaining counters here (timing hazard).
                            // Header fields are being latched in this cycle; reading them immediately
                            // in same cycle causes combinational loop. Deferred to RX_ROW_PTR_INIT, etc.
                        end else begin
                            rx_byte_count <= rx_byte_count + 1;
                            // request next byte if FIFO has data
                            if (fifo_count > 0) fifo_get <= 1'b1; else fifo_get <= 1'b0;
                        end
                    end else begin
                        // If a registered FIFO byte is not present but memory has data, request a read
                        if (fifo_count > 0) begin
                            fifo_get <= 1'b1;
                        end else begin
                            fifo_get <= 1'b0;
                        end
                    end
                end

                RX_ROW_PTR_INIT: begin
                    // Initialize row_ptr receive phase. Safe to use hdr_num_block_rows here because
                    // it was latched during RX_HDR_DATA in the *previous* cycle and is now stable.
                    rx_word_buf <= 32'd0;
                    rx_byte_count <= 2'd0;
                    row_ptr_remaining <= hdr_num_block_rows;
                end
                
                RX_ROW_PTR_DATA: begin
                    if (fifo_q_valid) begin
                        fifo_consume <= 1'b1;
                        // Place the registered FIFO byte into the correct 8-bit slice
                        if (rx_byte_count == 2'd3) begin
                            // Construct final word directly from registered MSB and previous bytes
                            assembled_word <= {fifo_q, rx_word_buf[23:0]};
                            row_ptr_wdata <= {fifo_q, rx_word_buf[23:0]};
                            row_ptr_waddr <= row_ptr_addr;
                            row_ptr_we <= 1'b1;
                            row_ptr_addr <= row_ptr_addr + 1;
                            rx_byte_count <= 2'd0;
                            rx_word_buf <= 32'd0;
                            if (                            black --check --line-length 120 "ACCEL-v1/accel v1/python/" > 0) row_ptr_remaining <= row_ptr_remaining - 1;
                            // move to next stage if done
                            if (row_ptr_remaining == 1) next_state = RX_COL_IDX_INIT;
                        end else begin
                            // Assign the 8-bit byte to its slot in rx_word_buf
                            rx_word_buf[rx_byte_count * 8 +: 8] <= fifo_q;
                            rx_byte_count <= rx_byte_count + 1;
                        end
                    end else if (fifo_count > 0) begin
                        // request synchronous load of FIFO head into fifo_q
                        fifo_get <= 1'b1;
                    end else begin
                        fifo_get <= 1'b0;
                        fifo_consume <= 1'b0;
                    end
                end

                RX_COL_IDX_INIT: begin
                    // Initialize col_idx receive phase. hdr_total_blocks is stable from prior cycle.
                    rx_halfword <= 16'd0;
                    rx_byte_count <= 2'd0;
                    col_idx_remaining <= hdr_total_blocks;
                end
                
                RX_COL_IDX_DATA: begin
                    if (fifo_q_valid) begin
                        fifo_consume <= 1'b1;
                        if (rx_byte_count == 2'd1) begin
                            // Construct final 16-bit value from MSB + previously stored LSB
                            col_idx_wdata <= {fifo_q, rx_halfword[7:0]};
                            col_idx_waddr <= col_idx_addr;
                            col_idx_we <= 1'b1;
                            col_idx_addr <= col_idx_addr + 1;
                            rx_byte_count <= 2'd0;
                            rx_halfword <= 16'd0;
                            if (col_idx_remaining > 0) col_idx_remaining <= col_idx_remaining - 1;
                            if (col_idx_remaining == 1) next_state = RX_BLOCK_INIT;
                        end else begin
                            // Store LSB in lower 8 bits of rx_halfword
                            rx_halfword[rx_byte_count * 8 +: 8] <= fifo_q;
                            rx_byte_count <= rx_byte_count + 1;
                        end
                    end else if (fifo_count > 0) begin
                        fifo_get <= 1'b1;
                    end else begin
                        fifo_get <= 1'b0;
                        fifo_consume <= 1'b0;
                    end
                end
                
                RX_BLOCK_DATA: begin
                    if (fifo_q_valid) begin
                        fifo_consume <= 1'b1;
                        // Load bytes into word buffer; when a full 32-bit word assembled, enqueue in word FIFO
                        rx_word_buf[rx_byte_count * 8 +: 8] <= fifo_q;
                        block_byte_count <= block_byte_count + 1;
                        if (rx_byte_count == 2'd3) begin
                            // assembled 32-bit word ready (LSB-first order)
                            assembled_word <= {fifo_q, rx_word_buf[23:0]};
                            // enqueue assembled_word into the block write FIFO if space available
                            if (!block_write_fifo_full) begin
                                block_write_fifo_mem[block_write_fifo_wr_ptr] <= {fifo_q, rx_word_buf[23:0]};
                                block_write_fifo_wr_ptr <= block_write_fifo_wr_ptr + 1;
                                block_write_fifo_count <= block_write_fifo_count + 1;
                            end else begin
                                // FIFO full: set error (backpressure not supported yet)
                                dma_error <= 1'b1;
                                $display("%0t: ERROR: Block write FIFO full while receiving data", $time);
                            end
                            rx_byte_count <= 2'd0;
                            rx_word_buf <= 32'd0;
                        end else begin
                            rx_byte_count <= rx_byte_count + 1;
                        end
                        
                        // After 64 bytes, block is complete
                            if (block_byte_count == 32'd63) begin
                            block_byte_count <= 32'd0;
                            current_block_idx <= current_block_idx + 1;
                            blocks_in_layer <= blocks_in_layer + 1;
                            blocks_written <= blocks_written + 1;
                            if (block_remaining > 0) block_remaining <= block_remaining - 1;
                            // end condition when all blocks received
                            if (block_remaining == 1) begin
                                // If CRC enabled, go read the host-supplied CRC next
                                if (ENABLE_CRC == 1) begin
                                    next_state = RX_CRC_CHK;
                                    // Reset per-CRC-byte counters/assembler
                                    rx_byte_count <= 2'd0;
                                    host_crc <= 32'd0;
                                end else begin
                                    next_state = TX_STATUS;
                                end
                            end
                        end
                    end else if (fifo_count > 0) begin
                        fifo_get <= 1'b1;
                    end else begin
                        fifo_get <= 1'b0;
                        fifo_consume <= 1'b0;
                    end
                end

                RX_CRC_CHK: begin
                    // Collect host CRC LSB-first (4 bytes) into host_crc
                    if (fifo_q_valid) begin
                        fifo_consume <= 1'b1;
                        host_crc[rx_byte_count * 8 +: 8] <= fifo_q;
                        if (rx_byte_count == 2'd3) begin
                            // Compare assembled host CRC to inverted computed CRC
                            if (host_crc != (~crc_accumulator)) begin
                                dma_error <= 1'b1;
                                $display("%0t: CRC mismatch: expected 0x%08x got 0x%08x", $time, ~crc_accumulator, host_crc);
                            end
                            // Reset CRC assembler and counters
                            rx_byte_count <= 2'd0;
                            host_crc <= 32'd0;
                            rx_word_buf <= 32'd0;
                            next_state <= TX_STATUS;
                        end else begin
                            rx_byte_count <= rx_byte_count + 1;
                        end
                    end else if (fifo_count > 0) begin
                        fifo_get <= 1'b1;
                    end else begin
                        fifo_get <= 1'b0;
                        fifo_consume <= 1'b0;
                    end
                end
                
                TX_STATUS: begin
                    if (uart_tx_ready) begin
                        uart_tx_data <= {6'd0, dma_error, dma_done};
                        uart_tx_valid <= 1'b1;
                    end
                end
                
                DONE: begin
                    dma_busy <= 1'b0;
                    dma_done <= 1'b1;
                end
            endcase
        end
    end
// FIFO write/pop logic and handling of write_en/read_en
    wire uart_rx_write_en = uart_rx_valid && uart_rx_ready;

    // Synchronous FIFO (registered read into fifo_q)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            fifo_wr_ptr <= {FIFO_PTR_W{1'b0}};
            fifo_rd_ptr <= {FIFO_PTR_W{1'b0}};
            fifo_count <= {FIFO_PTR_W+1{1'b0}};
            fifo_q <= 8'd0;
            fifo_q_valid <= 1'b0;
        end else begin
            // write to FIFO when host provides data and DUT accepts
            if (uart_rx_write_en) begin
                fifo_mem[fifo_wr_ptr] <= uart_rx_data;
                fifo_wr_ptr <= fifo_wr_ptr + 1;
            end

            // If requested and no current fifo_q present, load registered Q from memory
            if (fifo_get && !fifo_q_valid && (fifo_count > 0)) begin
                fifo_q <= fifo_mem[fifo_rd_ptr];
                fifo_rd_ptr <= fifo_rd_ptr + 1;
                fifo_q_valid <= 1'b1;
            end

            // If the FSM consumes the registered q, clear the valid flag and update CRC (only for payload bytes)
            if (fifo_consume) begin
                if (!fifo_q_valid) begin
                    // Underflow: attempt to consume when no registered data
                    dma_error <= 1'b1;
                    $display("%0t: ERROR: FIFO consume requested when no valid Q", $time);
                end else begin
                    // CRC is accumulated *only* during RX_BLOCK_DATA phase (payload bytes).
                    // Header fields (row_ptr, col_idx) are NOT included in CRC per protocol design:
                    // only the 64-byte block payloads are protected. Metadata is assumed reliable/trusted.
                    if (ENABLE_CRC == 1 && (state == RX_BLOCK_DATA)) begin
                        crc_accumulator <= crc32_byte(crc_accumulator, fifo_q);
                    end
                end
                fifo_q_valid <= 1'b0;
            end

            // Update count: write increments, memory-read decrements (pop from memory into fifo_q)
            // write only -> +1; read_mem only -> -1; both -> 0
            case ({uart_rx_write_en, (fifo_get && (fifo_count > 0))})
                2'b10: fifo_count <= fifo_count + 1;
                2'b01: fifo_count <= fifo_count - 1;
                2'b11: fifo_count <= fifo_count; // write and read in same cycle
                default: fifo_count <= fifo_count;
            endcase
        end
    end
    
    // ========================================================================
    // FSM Next State Logic
    // ========================================================================
    always @(*) begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (csr_wen && csr_addr == CSR_DMA_CTRL && csr_wdata[0]) begin
                    next_state = RX_LAYER_CMD;
                end
            end
            
            RX_LAYER_CMD: begin
                if (uart_rx_valid || (fifo_count > 0)) begin
                    next_state = RX_HDR_INIT;
                end
            end
            
            RX_ROW_PTR_INIT: begin
                // Could receive N row_ptr entries; simplified: go directly to receive
                next_state = RX_ROW_PTR_DATA;
            end
            
            RX_ROW_PTR_DATA: begin
                // Continue until CSR command changes (TODO: protocol refinement)
                // For now, receive a fixed number or detect end-of-data
                // Simplified: move to col_idx after some internal timeout
            end
            
            RX_COL_IDX_INIT: begin
                next_state = RX_COL_IDX_DATA;
            end
            
            RX_COL_IDX_DATA: begin
                // Continue receiving col_idx entries
            end
            
            RX_BLOCK_INIT: begin
                // Compute word-based start address for block data
                // Each block is BLOCK_SIZE bytes; in word address there are BLOCK_SIZE/WORD_BYTES words per block
                block_addr_word <= current_block_idx * (BLOCK_SIZE / WORD_BYTES);
                next_state = RX_BLOCK_DATA;
            end
            
            RX_BLOCK_DATA: begin
                // Continue until block_byte_count reaches MAX_BLOCKS or end signal
            end
            
            TX_STATUS: begin
                if (uart_tx_ready) begin
                    next_state = DONE;
                end
            end

            RX_CRC_CHK: begin
                // Stay in CRC check until 4 bytes are collected and consumed
                if (fifo_q_valid) begin
                    if (rx_byte_count == 2'd3) begin
                        // Byte count will be checked/consumed in sequential logic; move on
                        next_state = TX_STATUS;
                    end else begin
                        next_state = RX_CRC_CHK;
                    end
                end else if (fifo_count > 0) begin
                    next_state = RX_CRC_CHK;
                end else begin
                    next_state = RX_CRC_CHK;
                end
            end
            
            DONE: begin
                if (csr_wen && csr_addr == CSR_DMA_CTRL && csr_wdata[1]) begin
                    // Reset flag clears DMA
                    next_state = IDLE;
                end
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    // ========================================================================
    // CSR Read Path
    // ========================================================================
    always @(*) begin
        csr_rdata = 32'h0000_0000;
        
        case (csr_addr)
            CSR_DMA_LAYER: begin
                csr_rdata = {29'd0, active_layer};
            end
            CSR_DMA_COUNT: begin
                csr_rdata = blocks_written;
            end
            CSR_DMA_STATUS: begin
                csr_rdata = {29'd0, dma_error, dma_done, dma_busy};
            end
            CSR_DMA_BURST: begin
                csr_rdata = {31'd0, burst_enable};
            end
            default: csr_rdata = 32'hDEAD_BEEF;
        endcase
    end
    
    // ========================================================================
    // Default Output Assignments (Combinational)
    // ========================================================================
    // Accept bytes in RX-capable states, when input FIFO is not full, AND when block-write FIFO has space.
    // This prevents data loss by halting the host when either queue would overflow (backpressure).
    assign uart_rx_ready = ((state == RX_LAYER_CMD) || 
                           (state == RX_HDR_DATA) || 
                           (state == RX_ROW_PTR_DATA) || 
                           (state == RX_COL_IDX_DATA) || 
                           (state == RX_BLOCK_DATA)) && 
                          !fifo_full && 
                          (block_write_fifo_count < (BLOCK_WRITE_FIFO_DEPTH - 1));

endmodule

`default_nettype wire

// =============================================================================
// End of bsr_dma.v
// =============================================================================
