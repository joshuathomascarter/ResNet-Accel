// =============================================================================
// tb_axi_lite_slave.sv — Comprehensive Testbench for AXI4-Lite Slave
// =============================================================================
// Simplified for iverilog compatibility (pure Verilog-2001 constructs)
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_axi_lite_slave;

    // ========================================================================
    // Parameters
    // ========================================================================
    localparam CSR_ADDR_WIDTH = 8;
    localparam CSR_DATA_WIDTH = 32;
    localparam CLK_PERIOD = 10;
    
    // CSR Address Map
    localparam [7:0] CSR_DMA_LAYER  = 8'h50;
    localparam [7:0] CSR_DMA_CTRL   = 8'h51;
    localparam [7:0] CSR_DMA_COUNT  = 8'h52;
    localparam [7:0] CSR_DMA_STATUS = 8'h53;
    localparam [7:0] CSR_DMA_BURST  = 8'h54;
    localparam [7:0] INVALID_ADDR   = 8'hFF;
    
    // Response codes
    localparam [1:0] RESP_OKAY   = 2'b00;
    localparam [1:0] RESP_SLVERR = 2'b11;
    
    // ========================================================================
    // Signals
    // ========================================================================
    reg clk, rst_n;
    reg [CSR_ADDR_WIDTH-1:0] s_axi_awaddr;
    reg [2:0] s_axi_awprot;
    reg s_axi_awvalid;
    wire s_axi_awready;
    reg [CSR_DATA_WIDTH-1:0] s_axi_wdata;
    reg [(CSR_DATA_WIDTH/8)-1:0] s_axi_wstrb;
    reg s_axi_wvalid;
    wire s_axi_wready;
    wire [1:0] s_axi_bresp;
    wire s_axi_bvalid;
    reg s_axi_bready;
    reg [CSR_ADDR_WIDTH-1:0] s_axi_araddr;
    reg [2:0] s_axi_arprot;
    reg s_axi_arvalid;
    wire s_axi_arready;
    wire [CSR_DATA_WIDTH-1:0] s_axi_rdata;
    wire [1:0] s_axi_rresp;
    wire s_axi_rvalid;
    reg s_axi_rready;
    wire [CSR_ADDR_WIDTH-1:0] csr_addr;
    wire csr_wen, csr_ren;
    wire [CSR_DATA_WIDTH-1:0] csr_wdata;
    wire axi_error;
    
    // Test tracking
    integer test_count, test_pass, test_fail;
    integer i, j, k, cycles;
    
    // CSR memory (for loopback simulation)
    reg [CSR_DATA_WIDTH-1:0] csr_mem [0:255];
    reg [CSR_DATA_WIDTH-1:0] csr_rdata_sim;
    
    // ========================================================================
    // DUT
    // ========================================================================
    axi_lite_slave #(
        .CSR_ADDR_WIDTH(CSR_ADDR_WIDTH),
        .CSR_DATA_WIDTH(CSR_DATA_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awprot(s_axi_awprot),
        .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arprot(s_axi_arprot),
        .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .csr_addr(csr_addr), .csr_wen(csr_wen), .csr_ren(csr_ren),
        .csr_wdata(csr_wdata), .csr_rdata(csr_rdata_sim), .axi_error(axi_error)
    );
    
    // ========================================================================
    // Clock & Reset
    // ========================================================================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    initial begin
        rst_n = 0;
        #(CLK_PERIOD * 2) rst_n = 1;
    end
    
    // ========================================================================
    // CSR Loopback Simulation
    // ========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 256; i = i + 1) begin
                csr_mem[i] <= 32'd0;
            end
        end else begin
            // Write
            if (csr_wen) begin
                csr_mem[csr_addr] <= csr_wdata;
                $display("[CSR-W] STORED addr=0x%02x data=0x%08x", csr_addr, csr_wdata);
            end
        end
    end
    
    // Combinational read
    always @(*) begin
        csr_rdata_sim = csr_mem[csr_addr];
        if (csr_ren) begin
            $display("[CSR-R] FETCHED addr=0x%02x data=0x%08x", csr_addr, csr_mem[csr_addr]);
        end
    end
    
    // ========================================================================
    // Helper: Write Single CSR
    // ========================================================================
    task write_single;
        input [7:0] addr;
        input [31:0] data;
        input [1:0] exp_resp;
    begin
        test_count = test_count + 1;
        $display("\n[TEST %0d] WRITE addr=0x%02x data=0x%08x", test_count, addr, data);
        
        // Pulse AW+W together; slave should accept immediately (ready=1 by default)
        s_axi_awaddr = addr;
        s_axi_awvalid = 1'b1;
        s_axi_wdata = data;
        s_axi_wvalid = 1'b1;
        s_axi_wstrb = 4'b1111;
        s_axi_bready = 1'b1;
        @(posedge clk);
        
        $display("  [INFO] AW/W sent, waiting for response");
        
        // Response should arrive within 3 cycles
        cycles = 0;
        while (!s_axi_bvalid && cycles < 10) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        
        if (s_axi_bvalid) begin
            if (s_axi_bresp == exp_resp) begin
                $display("  [PASS] Response=%b", s_axi_bresp);
                test_pass = test_pass + 1;
            end else begin
                $display("  [FAIL] Response=%b (expected %b)", s_axi_bresp, exp_resp);
                test_fail = test_fail + 1;
            end
        end else begin
            $display("  [FAIL] No response");
            test_fail = test_fail + 1;
        end
        
        s_axi_awvalid = 1'b0;
        s_axi_wvalid = 1'b0;
        s_axi_bready = 1'b0;
        @(posedge clk);
    end
    endtask
    
    // ========================================================================
    // Helper: Read Single CSR
    // ========================================================================
    task read_single;
        input [7:0] addr;
        input [31:0] exp_data;
        input [1:0] exp_resp;
    begin
        test_count = test_count + 1;
        $display("\n[TEST %0d] READ  addr=0x%02x", test_count, addr);
        
        // Send AR; slave should accept immediately
        s_axi_araddr = addr;
        s_axi_arvalid = 1'b1;
        s_axi_rready = 1'b1;
        @(posedge clk);
        
        $display("  [INFO] AR sent, waiting for data");
        
        // Data should arrive within 3 cycles
        cycles = 0;
        while (!s_axi_rvalid && cycles < 10) begin
            @(posedge clk);
            cycles = cycles + 1;
        end
        
        if (s_axi_rvalid) begin
            if (s_axi_rresp == exp_resp) begin
                if (exp_resp == RESP_SLVERR) begin
                    $display("  [PASS] Error response");
                    test_pass = test_pass + 1;
                end else begin
                    if (s_axi_rdata == exp_data) begin
                        $display("  [PASS] Data=0x%08x", s_axi_rdata);
                        test_pass = test_pass + 1;
                    end else begin
                        $display("  [FAIL] Data=0x%08x (exp 0x%08x)", s_axi_rdata, exp_data);
                        test_fail = test_fail + 1;
                    end
                end
            end else begin
                $display("  [FAIL] Resp=%b (exp %b)", s_axi_rresp, exp_resp);
                test_fail = test_fail + 1;
            end
        end else begin
            $display("  [FAIL] No data");
            test_fail = test_fail + 1;
        end
        
        s_axi_arvalid = 1'b0;
        s_axi_rready = 1'b0;
        @(posedge clk);
    end
    endtask
    
    // ========================================================================
    // Main Test Sequence
    // ========================================================================
    initial begin
        test_count = 0;
        test_pass = 0;
        test_fail = 0;
        
        // Initialize signals
        s_axi_awaddr = 0;   s_axi_awvalid = 0;
        s_axi_wdata = 0;    s_axi_wvalid = 0;   s_axi_bready = 0;
        s_axi_araddr = 0;   s_axi_arvalid = 0;  s_axi_rready = 0;
        
        wait(rst_n);
        @(posedge clk);
        
        $display("\n===========================================");
        $display("  AXI4-Lite Slave Testbench");
        $display("===========================================\n");
        
        // Suite 1: Valid writes to CSR registers
        $display(">>> Suite 1: Single Writes (Valid Addresses)\n");
        write_single(CSR_DMA_LAYER,  32'h0000_0001, RESP_OKAY);
        write_single(CSR_DMA_CTRL,   32'h0000_0001, RESP_OKAY);
        write_single(CSR_DMA_COUNT,  32'hDEAD_BEEF, RESP_OKAY);
        
        // Suite 2: Read back written values
        $display("\n>>> Suite 2: Single Reads (Verify Writes)\n");
        read_single(CSR_DMA_LAYER,  32'h0000_0001, RESP_OKAY);
        read_single(CSR_DMA_CTRL,   32'h0000_0001, RESP_OKAY);
        read_single(CSR_DMA_COUNT,  32'hDEAD_BEEF, RESP_OKAY);
        
        // Suite 3: Invalid addresses (error responses)
        $display("\n>>> Suite 3: Invalid Addresses (Error Responses)\n");
        write_single(INVALID_ADDR, 32'h1111_1111, RESP_SLVERR);
        read_single(INVALID_ADDR,  32'h0000_0000, RESP_SLVERR);
        
        // Suite 4: Edge cases (all zeros, all ones)
        $display("\n>>> Suite 4: Edge Cases\n");
        write_single(CSR_DMA_LAYER, 32'h0000_0000, RESP_OKAY);
        read_single(CSR_DMA_LAYER,  32'h0000_0000, RESP_OKAY);
        write_single(CSR_DMA_CTRL,  32'hFFFF_FFFF, RESP_OKAY);
        read_single(CSR_DMA_CTRL,   32'hFFFF_FFFF, RESP_OKAY);
        
        // Final report
        @(posedge clk);
        @(posedge clk);
        $display("\n===========================================");
        $display("  TEST SUMMARY");
        $display("===========================================");
        $display("  Total:  %0d", test_count);
        $display("  Passed: %0d", test_pass);
        $display("  Failed: %0d", test_fail);
        if (test_fail == 0) begin
            $display("  Status: ALL PASSED");
        end else begin
            $display("  Status: SOME FAILED");
        end
        $display("===========================================\n");
        
        $finish;
    end
    
    // Timeout guard
    initial begin
        #(CLK_PERIOD * 100000);
        $display("\n[ERROR] TIMEOUT");
        $finish;
    end

endmodule

`default_nettype wire
