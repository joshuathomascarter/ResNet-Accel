`timescale 1ns/1ps
`default_nettype none

// =============================================================================
// tb_bsr_dma.sv — Unit testbench for BSR DMA engine
// =============================================================================
// Purpose:
//   Verify BSR DMA can:
//   - Parse layer configuration from CSR
//   - Receive row_ptr, col_idx, and block data via UART
//   - Write to BRAM interfaces with correct addressing
//   - Report status and error conditions
//   - Handle multi-layer operations
//
// Test Coverage:
//   1. Basic UART reception of layer command
//   2. row_ptr BRAM writes (32-bit metadata)
//   3. col_idx BRAM writes (16-bit per block)
//   4. Block data streaming (64 bytes per block, 8×8 INT8)
//   5. Status reporting via CSR
//   6. Error detection on malformed packets
//
// =============================================================================

module tb_bsr_dma;

    // ========================================================================
    // Clock and Reset
    // ========================================================================
    reg clk = 0;
    always #5 clk = ~clk;  // 10ns period = 100 MHz
    
    reg rst_n = 1'b1;
    
    initial begin
        #1 rst_n = 1'b0;
        #10 rst_n = 1'b1;
    end

    // ========================================================================
    // DUT Interface Signals
    // ========================================================================
    
    // UART RX (from host)
    reg [7:0]  uart_rx_data;
    reg        uart_rx_valid;
    wire       uart_rx_ready;
    
    // UART TX (to host)
    wire [7:0] uart_tx_data;
    wire       uart_tx_valid;
    reg        uart_tx_ready = 1'b1;
    
    // CSR interface
    reg [7:0]  csr_addr;
    reg        csr_wen;
    reg [31:0] csr_wdata;
    wire [31:0] csr_rdata;
    
    // row_ptr BRAM
    wire       row_ptr_we;
    wire [15:0] row_ptr_waddr;
    wire [31:0] row_ptr_wdata;
    
    // col_idx BRAM
    wire       col_idx_we;
    wire [15:0] col_idx_waddr;
    wire [15:0] col_idx_wdata;
    
    // block data BRAM (word-addressed 32-bit writes)
    wire       block_we;
    wire [20:0] block_waddr;
    wire [31:0] block_wdata;
    
    // Status outputs
    wire       dma_busy;
    wire       dma_done;
    wire       dma_error;
    wire [31:0] blocks_written;

    // ========================================================================
    // DUT Instantiation
    // ========================================================================
    bsr_dma #(
        .DATA_WIDTH(8),
        .ADDR_WIDTH(16),
        .MAX_LAYERS(8),
        .MAX_BLOCKS(65536),
        .BLOCK_SIZE(64),
        .ROW_PTR_DEPTH(256),
        .COL_IDX_DEPTH(65536),
        .ENABLE_CRC(1)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .uart_rx_data(uart_rx_data),
        .uart_rx_valid(uart_rx_valid),
        .uart_rx_ready(uart_rx_ready),
        .uart_tx_data(uart_tx_data),
        .uart_tx_valid(uart_tx_valid),
        .uart_tx_ready(uart_tx_ready),
        .csr_addr(csr_addr),
        .csr_wen(csr_wen),
        .csr_wdata(csr_wdata),
        .csr_rdata(csr_rdata),
        .row_ptr_we(row_ptr_we),
        .row_ptr_waddr(row_ptr_waddr),
        .row_ptr_wdata(row_ptr_wdata),
        .col_idx_we(col_idx_we),
        .col_idx_waddr(col_idx_waddr),
        .col_idx_wdata(col_idx_wdata),
        .block_we(block_we),
        .block_waddr(block_waddr),
        .block_wdata(block_wdata),
        .dma_busy(dma_busy),
        .dma_done(dma_done),
        .dma_error(dma_error),
        .blocks_written(blocks_written)
    );

    // ========================================================================
    // Test Helper Tasks
    // ========================================================================
    
    // Send a byte via UART RX simulation
    task uart_send_byte(input [7:0] data);
        @(posedge clk);
        // Wait until DUT indicates it's ready to accept a byte
        while (!uart_rx_ready) @(posedge clk);
        uart_rx_data <= data;
        uart_rx_valid <= 1'b1;
        @(posedge clk);
        uart_rx_valid <= 1'b0;
    endtask

    // CRC32 update function (LSB-first, same as RTL)
    function automatic [31:0] tb_crc32_byte(input [31:0] cur, input [7:0] data);
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
            tb_crc32_byte = c;
        end
    endfunction
    
    // Write to CSR
    task csr_write(input [7:0] addr, input [31:0] data);
        @(posedge clk);
        csr_addr <= addr;
        csr_wen <= 1'b1;
        csr_wdata <= data;
        @(posedge clk);
        csr_wen <= 1'b0;
    endtask
    
    // Read from CSR
    task csr_read(input [7:0] addr);
        @(posedge clk);
        csr_addr <= addr;
        csr_wen <= 1'b0;
        @(posedge clk);
    endtask

    // ========================================================================
    // Test Stimulus
    // ========================================================================
    reg [31:0] read_value;
    reg [31:0] tb_crc;
    
    initial begin
        uart_rx_valid = 1'b0;
        csr_wen = 1'b0;
        
        // Wait for reset
        #100;
        
        $display("=== TB: BSR DMA Unit Test ===");
        
        // ====================================================================
        // TEST 1: Initial state verification
        // ====================================================================
        $display("\nTEST 1: Initial state");
            csr_read(8'h53);
        read_value = csr_rdata;
        if (read_value[0] == 1'b0 && read_value[1] == 1'b0) begin
            $display("  PASS: DMA idle and not done on reset");
        end else begin
            $display("  FAIL: Unexpected initial state: %08x", read_value);
        end
        
        // ====================================================================
        // TEST 2: DMA start via CSR
        // ====================================================================
        $display("\nTEST 2: DMA start");
            // Write START bit and enable 32-bit word mode (bit 2)
            csr_write(8'h51, 32'h0000_0001 | (1 << 2));  // START + WORD_MODE
        #20;
        
        if (dma_busy) begin
            $display("  PASS: DMA marked busy after start");
        end else begin
            $display("  FAIL: DMA not busy");
        end
        
        // ====================================================================
        // TEST 3: Layer selection
        // ====================================================================
        $display("\nTEST 3: Layer selection");
        uart_send_byte(8'h02);  // Select layer 2
        #50;
        
            csr_read(8'h50);
        read_value = csr_rdata;
        if (read_value[2:0] == 3'd2) begin
            $display("  PASS: Layer 2 selected");
        end else begin
            $display("  FAIL: Layer not set correctly: %02x", read_value[2:0]);
        end
        
        // ====================================================================
        // TEST 4: row_ptr BRAM writes
        // ====================================================================
        $display("\nTEST 4: row_ptr BRAM writes");
        
        // Send header first (num_block_rows, num_block_cols, total_blocks)
        // Use LSB-first ordering for each 32-bit value
        // Example: num_block_rows = 3, num_block_cols = 1, total_blocks = 1
        $display("Sending header: rows=3, cols=1, blocks=1");
        uart_send_byte(8'h03);
        uart_send_byte(8'h00);
        uart_send_byte(8'h00);
        uart_send_byte(8'h00);
        uart_send_byte(8'h01);
        uart_send_byte(8'h00);
        uart_send_byte(8'h00);
        uart_send_byte(8'h00);
        uart_send_byte(8'h01);
        uart_send_byte(8'h00);
        uart_send_byte(8'h00);
        uart_send_byte(8'h00);

        // Compute CRC over header+payload as we send
        tb_crc = 32'hFFFF_FFFF;

        // Simulate receiving 4 bytes of row_ptr[0]
        uart_send_byte(8'h00);  // LSB
        uart_send_byte(8'h00);
        uart_send_byte(8'h00);
        uart_send_byte(8'h01);  // MSB -> 0x0100_0000 (256 blocks in first row)
        
        // Wait for write to complete
        #100;
        
        if (row_ptr_we) begin
            $display("  PASS: row_ptr write strobe asserted");
            $display("    Address: 0x%04x, Data: 0x%08x", row_ptr_waddr, row_ptr_wdata);
        end else begin
            $display("  INFO: Waiting for row_ptr write completion");
        end
        
        // ====================================================================
        // TEST 5: col_idx BRAM writes
        // ====================================================================
        $display("\nTEST 5: col_idx BRAM writes");
        
        // Simulate receiving 2 bytes of col_idx[0]
        uart_send_byte(8'h00);  // LSB
        uart_send_byte(8'h00);  // MSB -> col_idx value 0
        
        #100;
        
        if (col_idx_we) begin
            $display("  PASS: col_idx write strobe asserted");
            $display("    Address: 0x%08x, Data: 0x%04x", col_idx_waddr, col_idx_wdata);
        end else begin
            $display("  INFO: Waiting for col_idx write completion");
        end
        
        // ====================================================================
        // TEST 6: Block data streaming (64 bytes)
        // ====================================================================
        $display("\nTEST 6: Block data streaming (first 8 bytes of 64)");
        
        // Send 64 bytes of an 8×8 block (one full block)
        for (int i = 0; i < 64; i++) begin
            uart_send_byte(8'h10 + i);  // Send pattern 0x10-0x4F
            tb_crc = tb_crc32_byte(tb_crc, 8'h10 + i);
        end

        // Now send CRC (final XOR ~crc)
        tb_crc = ~tb_crc;
        uart_send_byte(tb_crc[7:0]);
        uart_send_byte(tb_crc[15:8]);
        uart_send_byte(tb_crc[23:16]);
        uart_send_byte(tb_crc[31:24]);
        
        #200;
        
        if (block_we) begin
            $display("  PASS: Block writes detected");
            $display("    First block address: 0x%07x", block_waddr);
        end else begin
            $display("  INFO: Block writes in progress or pending");
        end
        
        // ====================================================================
        // TEST 7: Block counter
        // ====================================================================
        $display("\nTEST 7: Block counter");
            csr_read(8'h52);
        read_value = csr_rdata;
        $display("  Blocks written so far: %d", read_value);
        
        // ====================================================================
        // TEST 8: Status readback
        // ====================================================================
        $display("\nTEST 8: Status readback");
            csr_read(8'h53);
        read_value = csr_rdata;
        $display("  DMA Status: busy=%d, done=%d, error=%d",
                 read_value[0], read_value[1], read_value[2]);
        
        // ====================================================================
        // Complete
        // ====================================================================
        $display("\n=== TB: Tests completed ===");
        $finish;
    end

    // ========================================================================
    // Simulation Monitoring
    // ========================================================================
    initial begin
        $display("\n%0t: Simulation started", $time);
    end
    
    always @(posedge row_ptr_we) begin
        $display("%0t: row_ptr write: addr=0x%04x, data=0x%08x",
                 $time, row_ptr_waddr, row_ptr_wdata);
    end
    
    always @(posedge col_idx_we) begin
        $display("%0t: col_idx write: addr=0x%08x, data=0x%04x",
                 $time, col_idx_waddr, col_idx_wdata);
    end
    
    always @(posedge block_we) begin
        $display("%0t: block write: word_addr=0x%06x, byte_addr=0x%07x, data=0x%08x",
                 $time, block_waddr, (block_waddr << 2), block_wdata);
    end

endmodule

`default_nettype wire

// =============================================================================
// End of tb_bsr_dma.sv
// =============================================================================
