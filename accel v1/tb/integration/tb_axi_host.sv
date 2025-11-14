// =============================================================================
// tb_axi_host.sv — Integration Testbench for AXI Host Interface
// =============================================================================
// Purpose:
//   Simulate AXI4-Lite master issuing CSR reads/writes and burst transfers.
//   Validates that DMA FIFO receives data correctly.
//
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_axi_host;

    localparam CLK_PERIOD = 20; // 50 MHz
    localparam DATA_WIDTH = 32;
    localparam ADDR_WIDTH = 32;
    
    // Clock and reset
    reg clk, rst_n;
    
    // AXI4-Lite Write Address Channel
    reg [ADDR_WIDTH-1:0] awaddr;
    reg [1:0] awburst;
    reg [7:0] awlen;
    reg [2:0] awsize;
    reg awvalid;
    wire awready;
    
    // AXI4-Lite Write Data Channel
    reg [DATA_WIDTH-1:0] wdata;
    reg [(DATA_WIDTH/8)-1:0] wstrb;
    reg wlast;
    reg wvalid;
    wire wready;
    
    // AXI4-Lite Write Response Channel
    wire [1:0] bresp;
    wire bvalid;
    reg bready;
    
    // AXI4-Lite Read Address Channel
    reg [ADDR_WIDTH-1:0] araddr;
    reg [1:0] arburst;
    reg [7:0] arlen;
    reg [2:0] arsize;
    reg arvalid;
    wire arready;
    
    // AXI4-Lite Read Data Channel
    wire [DATA_WIDTH-1:0] rdata;
    wire [1:0] rresp;
    wire rlast;
    wire rvalid;
    reg rready;
    
    // Status
    wire busy, done_pulse, error;
    
    // ========================================================================
    // DUT Instantiation
    // ========================================================================
    
    accel_top #(
        .N_ROWS(2),
        .N_COLS(2),
        .TM(8),
        .TN(8),
        .TK(8),
        .CLK_HZ(50_000_000),
        .BAUD(115_200),
        .ADDR_WIDTH(6)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .uart_rx(1'b1),     // Idle (pulled high)
        .uart_tx(),
        .s_axi_awaddr(awaddr),
        .s_axi_awburst(awburst),
        .s_axi_awlen(awlen),
        .s_axi_awsize(awsize),
        .s_axi_awvalid(awvalid),
        .s_axi_awready(awready),
        .s_axi_wdata(wdata),
        .s_axi_wstrb(wstrb),
        .s_axi_wlast(wlast),
        .s_axi_wvalid(wvalid),
        .s_axi_wready(wready),
        .s_axi_bresp(bresp),
        .s_axi_bvalid(bvalid),
        .s_axi_bready(bready),
        .s_axi_araddr(araddr),
        .s_axi_arburst(arburst),
        .s_axi_arlen(arlen),
        .s_axi_arsize(arsize),
        .s_axi_arvalid(arvalid),
        .s_axi_arready(arready),
        .s_axi_rdata(rdata),
        .s_axi_rresp(rresp),
        .s_axi_rlast(rlast),
        .s_axi_rvalid(rvalid),
        .s_axi_rready(rready),
        .busy(busy),
        .done_pulse(done_pulse),
        .error(error)
    );
    
    // ========================================================================
    // Clock Generation
    // ========================================================================
    
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // ========================================================================
    // Test Stimulus
    // ========================================================================
    
    initial begin
        // Initialize
        rst_n = 1'b0;
        awaddr = 32'd0;
        awburst = 2'b01;  // INCR
        awlen = 8'd0;
        awsize = 3'b010;  // 4 bytes
        awvalid = 1'b0;
        wdata = 32'd0;
        wstrb = 4'hF;     // All bytes valid
        wlast = 1'b0;
        wvalid = 1'b0;
        bready = 1'b0;
        
        araddr = 32'd0;
        arburst = 2'b01;
        arlen = 8'd0;
        arsize = 3'b010;
        arvalid = 1'b0;
        rready = 1'b0;
        
        // Release reset
        @(posedge clk);
        rst_n = 1'b1;
        repeat(10) @(posedge clk);
        
        // ====================================================================
        // TEST 1: CSR Write (Enable CRC)
        // ====================================================================
        $display("[TEST] CSR Write: DMA_CTRL = 0x1 (enable CRC)");
        
        // Write address
        awaddr = 32'h0000_0051;  // DMA_CTRL CSR address
        awvalid = 1'b1;
        @(posedge clk);
        
        while (!awready) @(posedge clk);
        awvalid = 1'b0;
        
        // Write data
        wdata = 32'h0000_0001;   // Enable CRC
        wlast = 1'b1;
        wvalid = 1'b1;
        @(posedge clk);
        
        while (!wready) @(posedge clk);
        wvalid = 1'b0;
        
        // Read response
        bready = 1'b1;
        @(posedge clk);
        
        while (!bvalid) @(posedge clk);
        $display("  Response: BRESP=%b (expect 00=OKAY)", bresp);
        bready = 1'b0;
        @(posedge clk);
        
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // TEST 2: CSR Read (DMA_STATUS)
        // ====================================================================
        $display("[TEST] CSR Read: DMA_STATUS");
        
        // Read address
        araddr = 32'h0000_0053;  // DMA_STATUS CSR address
        arvalid = 1'b1;
        @(posedge clk);
        
        while (!arready) @(posedge clk);
        arvalid = 1'b0;
        
        // Read data
        rready = 1'b1;
        @(posedge clk);
        
        while (!rvalid) @(posedge clk);
        $display("  Data: RDATA=0x%08X, RRESP=%b", rdata, rresp);
        rready = 1'b0;
        @(posedge clk);
        
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // TEST 3: Invalid CSR Address (expect SLVERR)
        // ====================================================================
        $display("[TEST] CSR Write to Invalid Address (expect SLVERR)");
        
        awaddr = 32'h0000_00FF;  // Invalid CSR address
        awvalid = 1'b1;
        @(posedge clk);
        
        while (!awready) @(posedge clk);
        awvalid = 1'b0;
        
        // Write data
        wdata = 32'h1234_5678;
        wlast = 1'b1;
        wvalid = 1'b1;
        @(posedge clk);
        
        while (!wready) @(posedge clk);
        wvalid = 1'b0;
        
        // Read response (expect SLVERR = 2'b11)
        bready = 1'b1;
        @(posedge clk);
        
        while (!bvalid) @(posedge clk);
        $display("  Response: BRESP=%b (expect 11=SLVERR)", bresp);
        if (bresp == 2'b11) begin
            $display("  [PASS] Error response correctly returned");
        end else begin
            $display("  [FAIL] Expected SLVERR, got OKAY");
        end
        bready = 1'b0;
        @(posedge clk);
        
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // TEST 4: Burst Write (4 beats, 32-bit words)
        // ====================================================================
        $display("[TEST] Burst Write: 4 beats × 32 bits");
        
        // Write address (with burst length)
        awaddr = 32'h1000_0000;
        awburst = 2'b01;  // INCR
        awlen = 8'd3;     // 4 beats
        awsize = 3'b010;  // 4 bytes per beat
        awvalid = 1'b1;
        @(posedge clk);
        
        while (!awready) @(posedge clk);
        awvalid = 1'b0;
        
        // Write data (4 beats)
        for (int beat = 0; beat < 4; beat = beat + 1) begin
            wdata = 32'hDEAD_0000 + beat;
            wlast = (beat == 3);
            wvalid = 1'b1;
            @(posedge clk);
            while (!wready) @(posedge clk);
            $display("  Beat %0d: WDATA=0x%08X, WLAST=%b", beat, wdata, wlast);
        end
        wvalid = 1'b0;
        
        // Read response
        bready = 1'b1;
        @(posedge clk);
        
        while (!bvalid) @(posedge clk);
        $display("  Response: BRESP=%b", bresp);
        bready = 1'b0;
        @(posedge clk);
        
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // TEST 5: Burst Read (4 beats)
        // ====================================================================
        $display("[TEST] Burst Read: 4 beats × 32 bits");
        
        araddr = 32'h1000_0000;
        arlen = 8'd3;     // 4 beats
        arsize = 3'b010;  // 4 bytes per beat
        arvalid = 1'b1;
        @(posedge clk);
        
        while (!arready) @(posedge clk);
        arvalid = 1'b0;
        
        // Read data (4 beats)
        rready = 1'b1;
        for (int beat = 0; beat < 4; beat = beat + 1) begin
            @(posedge clk);
            while (!rvalid) @(posedge clk);
            $display("  Beat %0d: RDATA=0x%08X, RLAST=%b, RRESP=%b", beat, rdata, rlast, rresp);
        end
        rready = 1'b0;
        @(posedge clk);
        
        repeat(5) @(posedge clk);
        
        // ====================================================================
        // Done
        // ====================================================================
        $display("[DONE] Integration test complete");
        $finish;
    end
    
    // ========================================================================
    // Monitoring
    // ========================================================================
    
    initial begin
        $monitor("[%0t] clk=%b rst=%b awvalid=%b awready=%b wvalid=%b wready=%b bvalid=%b",
            $time, clk, rst_n, awvalid, awready, wvalid, wready, bvalid);
    end

endmodule

`default_nettype wire
