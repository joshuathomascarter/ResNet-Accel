// =============================================================================
// tb_axi_lite_slave_enhanced.sv — Enhanced AXI4-Lite Slave Testbench
// =============================================================================
// This enhanced version includes:
// - Improved documentation and readability
// - Verification of AXI protocol compliance
// - Better error reporting
// - Performance metrics
// - Configuration for both iverilog and Verilator
//
// Usage:
//   iverilog -g2009 -o tb.vvp axi_lite_slave.sv tb_axi_lite_slave_enhanced.sv
//   vvp tb.vvp
//
// Or with Verilator:
//   verilator -Wall --trace -cc axi_lite_slave.sv \
//     tb_axi_lite_slave_enhanced.sv --top tb_axi_lite_slave_enhanced
// =============================================================================

`timescale 1ns/1ps
`default_nettype none

module tb_axi_lite_slave_enhanced;

    // ========================================================================
    // Configuration
    // ========================================================================
    localparam string VERSION = "1.0";
    localparam string BUILD_DATE = "2025-11-16";
    localparam integer CLK_PERIOD_NS = 10;  // 100 MHz
    localparam integer RST_CYCLES = 2;      // Cycles to hold reset
    localparam integer TEST_TIMEOUT_US = 100;
    
    // CSR Configuration (must match DUT)
    localparam integer CSR_ADDR_WIDTH = 8;
    localparam integer CSR_DATA_WIDTH = 32;
    
    // CSR Address Map
    localparam [7:0] ADDR_DMA_LAYER   = 8'h50;
    localparam [7:0] ADDR_DMA_CTRL    = 8'h51;
    localparam [7:0] ADDR_DMA_COUNT   = 8'h52;
    localparam [7:0] ADDR_DMA_STATUS  = 8'h53;
    localparam [7:0] ADDR_DMA_BURST   = 8'h54;
    localparam [7:0] ADDR_INVALID     = 8'hFF;
    localparam [7:0] ADDR_OUT_OF_RANGE = 8'hA0;
    
    // AXI Response Codes
    localparam [1:0] RESP_OKAY   = 2'b00;  // Success
    localparam [1:0] RESP_EXOKAY = 2'b01;  // Exclusive access OK
    localparam [1:0] RESP_SLVERR = 2'b10;  // Slave error (e.g., invalid address)
    localparam [1:0] RESP_DECERR = 2'b11;  // Decode error
    
    // ========================================================================
    // Signals
    // ========================================================================
    
    // Clock and Reset
    reg clk, rst_n;
    
    // AXI Write Address Channel
    reg [CSR_ADDR_WIDTH-1:0] s_axi_awaddr;
    reg [2:0] s_axi_awprot;
    reg s_axi_awvalid;
    wire s_axi_awready;
    
    // AXI Write Data Channel
    reg [CSR_DATA_WIDTH-1:0] s_axi_wdata;
    reg [(CSR_DATA_WIDTH/8)-1:0] s_axi_wstrb;
    reg s_axi_wvalid;
    wire s_axi_wready;
    
    // AXI Write Response Channel
    wire [1:0] s_axi_bresp;
    wire s_axi_bvalid;
    reg s_axi_bready;
    
    // AXI Read Address Channel
    reg [CSR_ADDR_WIDTH-1:0] s_axi_araddr;
    reg [2:0] s_axi_arprot;
    reg s_axi_arvalid;
    wire s_axi_arready;
    
    // AXI Read Data Channel
    wire [CSR_DATA_WIDTH-1:0] s_axi_rdata;
    wire [1:0] s_axi_rresp;
    wire s_axi_rvalid;
    reg s_axi_rready;
    
    // CSR Interface (slave internal signals)
    wire [CSR_ADDR_WIDTH-1:0] csr_addr;
    wire csr_wen, csr_ren;
    wire [CSR_DATA_WIDTH-1:0] csr_wdata;
    wire axi_error;
    
    // Test tracking
    integer test_count = 0;
    integer test_pass = 0;
    integer test_fail = 0;
    integer total_write_cycles = 0;
    integer total_read_cycles = 0;
    
    // CSR memory (simulated slave storage)
    reg [CSR_DATA_WIDTH-1:0] csr_mem [0:255];
    reg [CSR_DATA_WIDTH-1:0] csr_rdata_sim;
    
    // Timing measurements
    real write_latency_ns[100];
    real read_latency_ns[100];
    integer write_latency_idx = 0;
    integer read_latency_idx = 0;
    real avg_write_latency = 0;
    real avg_read_latency = 0;
    
    // ========================================================================
    // DUT Instantiation
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
    // Clock Generation
    // ========================================================================
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD_NS/2) clk = ~clk;
    end
    
    // ========================================================================
    // Reset Generation
    // ========================================================================
    initial begin
        rst_n = 1'b0;
        repeat(RST_CYCLES) @(posedge clk);
        rst_n = 1'b1;
    end
    
    // ========================================================================
    // CSR Memory Simulation (Loopback)
    // ========================================================================
    // This simulates the backend register storage
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            integer i;
            for (i = 0; i < 256; i = i + 1) begin
                csr_mem[i] <= 32'd0;
            end
        end else if (csr_wen) begin
            csr_mem[csr_addr] <= csr_wdata;
            $display("[CSR_MEM-W] addr=0x%02x data=0x%08x @ %t",
                     csr_addr, csr_wdata, $realtime);
        end
    end
    
    // Combinational read  - must respond immediately to csr_addr
    always @(*) begin
        csr_rdata_sim = csr_mem[csr_addr];
        if (csr_ren) begin
            $display("[CSR_MEM-R] addr=0x%02x data=0x%08x @ %t",
                     csr_addr, csr_mem[csr_addr], $realtime);
        end
    end
    
    // ========================================================================
    // Utility: Response Code to String
    // ========================================================================
    function string resp_to_str(input [1:0] resp);
        case (resp)
            2'b00:   resp_to_str = "OKAY";
            2'b01:   resp_to_str = "EXOKAY";
            2'b10:   resp_to_str = "SLVERR";
            2'b11:   resp_to_str = "DECERR";
            default: resp_to_str = "UNKNOWN";
        endcase
    endfunction
    
    // ========================================================================
    // Task: Single AXI Write
    // ========================================================================
    task write_single;
        input [7:0] addr;
        input [31:0] data;
        input [1:0] exp_resp;
        output integer cycles;
        
        real start_time;
        begin
            test_count = test_count + 1;
            start_time = $realtime;
            
            $display("\n[TEST %0d] WRITE", test_count);
            $display("  addr=0x%02x, data=0x%08x, expect_resp=%s",
                     addr, data, resp_to_str(exp_resp));
            
            // Issue write request
            s_axi_awaddr = addr;
            s_axi_awvalid = 1'b1;
            s_axi_wdata = data;
            s_axi_wvalid = 1'b1;
            s_axi_wstrb = 4'b1111;  // All bytes valid
            s_axi_bready = 1'b1;
            @(posedge clk);
            
            // Wait for response (max 10 cycles)
            cycles = 0;
            while (!s_axi_bvalid && cycles < 10) begin
                @(posedge clk);
                cycles = cycles + 1;
            end
            
            if (s_axi_bvalid) begin
                real elapsed = ($realtime - start_time) / 1000.0;  // Convert to µs
                $display("  Response: %s (cycles=%0d, latency=%.2f µs)",
                         resp_to_str(s_axi_bresp), cycles, elapsed);
                
                if (s_axi_bresp == exp_resp) begin
                    $display("  ✓ PASS");
                    test_pass = test_pass + 1;
                    
                    // Record latency
                    if (write_latency_idx < 100) begin
                        write_latency_ns[write_latency_idx] = elapsed * 1000;
                        write_latency_idx = write_latency_idx + 1;
                    end
                    total_write_cycles = total_write_cycles + cycles;
                end else begin
                    $display("  ✗ FAIL: Expected %s, got %s",
                             resp_to_str(exp_resp), resp_to_str(s_axi_bresp));
                    test_fail = test_fail + 1;
                end
            end else begin
                $display("  ✗ FAIL: No response (timeout after %0d cycles)", cycles);
                test_fail = test_fail + 1;
            end
            
            // Deassert signals
            s_axi_awvalid = 1'b0;
            s_axi_wvalid = 1'b0;
            s_axi_bready = 1'b0;
            @(posedge clk);
            @(posedge clk);  // Extra cycle for settling
        end
    endtask
    
    // Task: Single AXI Read
    // ========================================================================
    task read_single;
        input [7:0] addr;
        input [31:0] exp_data;
        input [1:0] exp_resp;
        output integer cycles;
        
        real start_time;
        real elapsed;
        reg [CSR_DATA_WIDTH-1:0] captured_data;
        begin
            test_count = test_count + 1;
            start_time = $realtime;
            
            $display("\n[TEST %0d] READ", test_count);
            $display("  addr=0x%02x, expect_data=0x%08x, expect_resp=%s",
                     addr, exp_data, resp_to_str(exp_resp));
            
            // Issue read request
            s_axi_araddr = addr;
            s_axi_arvalid = 1'b1;
            s_axi_rready = 1'b1;
            @(posedge clk);
            
            // Wait for response (max 10 cycles)
            cycles = 0;
            while (!s_axi_rvalid && cycles < 10) begin
                @(posedge clk);
                cycles = cycles + 1;
            end
            
            // Capture data when response is valid
            if (s_axi_rvalid) begin
                captured_data = s_axi_rdata;
                elapsed = ($realtime - start_time) / 1000.0;  // Convert to µs
                $display("  Response: %s, data=0x%08x (cycles=%0d, latency=%.2f µs)",
                         resp_to_str(s_axi_rresp), captured_data, cycles, elapsed);
                
                if (s_axi_rresp == exp_resp) begin
                    if (exp_resp == RESP_SLVERR) begin
                        $display("  ✓ PASS (error response as expected)");
                        test_pass = test_pass + 1;
                    end else begin
                        if (captured_data == exp_data) begin
                            $display("  ✓ PASS (data matches)");
                            test_pass = test_pass + 1;
                        end else begin
                            $display("  ✗ FAIL: Data mismatch (expected 0x%08x, got 0x%08x)",
                                     exp_data, captured_data);
                            test_fail = test_fail + 1;
                        end
                    end
                    
                    // Record latency
                    if (read_latency_idx < 100) begin
                        read_latency_ns[read_latency_idx] = elapsed * 1000;
                        read_latency_idx = read_latency_idx + 1;
                    end
                    total_read_cycles = total_read_cycles + cycles;
                end else begin
                    $display("  ✗ FAIL: Response mismatch (expected %s, got %s)",
                             resp_to_str(exp_resp), resp_to_str(s_axi_rresp));
                    test_fail = test_fail + 1;
                end
            end else begin
                $display("  ✗ FAIL: No response (timeout after %0d cycles)", cycles);
                test_fail = test_fail + 1;
            end
            
            // Deassert signals
            s_axi_arvalid = 1'b0;
            s_axi_rready = 1'b0;
            @(posedge clk);
            @(posedge clk);
            @(posedge clk);  // Extra cycles for settling
        end
    endtask
    
    // ========================================================================
    // Main Test Sequence
    // ========================================================================
    initial begin
        integer cycles;
        integer i;
        real sum;
        
        // Initialize
        s_axi_awaddr = 0;   s_axi_awvalid = 0;   s_axi_awprot = 0;
        s_axi_wdata = 0;    s_axi_wvalid = 0;    s_axi_wstrb = 0;
        s_axi_bready = 0;
        s_axi_araddr = 0;   s_axi_arvalid = 0;   s_axi_arprot = 0;
        s_axi_rready = 0;
        
        // Wait for reset
        wait(rst_n);
        @(posedge clk);
        
        // Banner
        $display("\n");
        $display("╔════════════════════════════════════════════════════════════════╗");
        $display("║   AXI4-Lite Slave Enhanced Testbench (v%s)                 ║", VERSION);
        $display("║   Build Date: %s                                      ║", BUILD_DATE);
        $display("╚════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // ====================================================================
        // Test Suite 1: Valid Writes
        // ====================================================================
        $display("╭─ Test Suite 1: Valid Writes to CSR Registers");
        $display("│");
        write_single(ADDR_DMA_LAYER,  32'h0000_0001, RESP_OKAY, cycles);
        write_single(ADDR_DMA_CTRL,   32'h0000_0001, RESP_OKAY, cycles);
        write_single(ADDR_DMA_COUNT,  32'hDEAD_BEEF, RESP_OKAY, cycles);
        write_single(ADDR_DMA_STATUS, 32'h0000_0042, RESP_OKAY, cycles);
        $display("╰─ Suite 1 Complete");
        
        // Longer delay between suites to flush pipeline completely
        repeat(20) @(posedge clk);
        
        // ====================================================================
        // Test Suite 2: Read Back Verification
        // ====================================================================
        $display("\n╭─ Test Suite 2: Read Back & Verify Writes");
        $display("│");
        read_single(ADDR_DMA_LAYER,  32'h0000_0001, RESP_OKAY, cycles);
        read_single(ADDR_DMA_CTRL,   32'h0000_0001, RESP_OKAY, cycles);
        read_single(ADDR_DMA_COUNT,  32'hDEAD_BEEF, RESP_OKAY, cycles);
        read_single(ADDR_DMA_STATUS, 32'h0000_0042, RESP_OKAY, cycles);
        $display("╰─ Suite 2 Complete");
        
        // ====================================================================
        // Test Suite 3: Invalid Addresses (Error Responses)
        // ====================================================================
        $display("\n╭─ Test Suite 3: Invalid Addresses (Error Handling)");
        $display("│");
        write_single(ADDR_INVALID, 32'h1111_1111, RESP_SLVERR, cycles);
        read_single(ADDR_INVALID, 32'h0000_0000, RESP_SLVERR, cycles);
        write_single(ADDR_OUT_OF_RANGE, 32'hFFFF_FFFF, RESP_SLVERR, cycles);
        $display("╰─ Suite 3 Complete");
        
        // ====================================================================
        // Test Suite 4: Edge Cases
        // ====================================================================
        $display("\n╭─ Test Suite 4: Edge Cases");
        $display("│");
        write_single(ADDR_DMA_LAYER, 32'h0000_0000, RESP_OKAY, cycles);
        read_single(ADDR_DMA_LAYER, 32'h0000_0000, RESP_OKAY, cycles);
        write_single(ADDR_DMA_CTRL, 32'hFFFF_FFFF, RESP_OKAY, cycles);
        read_single(ADDR_DMA_CTRL, 32'hFFFF_FFFF, RESP_OKAY, cycles);
        write_single(ADDR_DMA_BURST, 32'hAA55_AA55, RESP_OKAY, cycles);
        read_single(ADDR_DMA_BURST, 32'hAA55_AA55, RESP_OKAY, cycles);
        $display("╰─ Suite 4 Complete");
        
        // ====================================================================
        // Final Report
        // ====================================================================
        @(posedge clk);
        @(posedge clk);
        
        // Calculate statistics
        if (write_latency_idx > 0) begin
            sum = 0;
            for (i = 0; i < write_latency_idx; i = i + 1) begin
                sum = sum + write_latency_ns[i];
            end
            avg_write_latency = sum / write_latency_idx;
        end
        
        if (read_latency_idx > 0) begin
            sum = 0;
            for (i = 0; i < read_latency_idx; i = i + 1) begin
                sum = sum + read_latency_ns[i];
            end
            avg_read_latency = sum / read_latency_idx;
        end
        
        $display("\n");
        $display("╔════════════════════════════════════════════════════════════════╗");
        $display("║   TEST SUMMARY                                                 ║");
        $display("╠════════════════════════════════════════════════════════════════╣");
        $display("║  Total Tests:    %4d                                           ║", test_count);
        $display("║  Passed:         %4d                                           ║", test_pass);
        $display("║  Failed:         %4d                                           ║", test_fail);
        if (test_fail == 0) begin
            $display("║  Status:         ✓ ALL PASSED                                ║");
        end else begin
            $display("║  Status:         ✗ FAILURES DETECTED                          ║");
        end
        $display("╠════════════════════════════════════════════════════════════════╣");
        $display("║  Performance Metrics:                                          ║");
        if (write_latency_idx > 0) begin
            $display("║    Write Transactions:  %0d", write_latency_idx);
            $display("║    Avg Write Latency:   %.2f ns                               ║",
                     avg_write_latency);
        end
        if (read_latency_idx > 0) begin
            $display("║    Read Transactions:   %0d", read_latency_idx);
            $display("║    Avg Read Latency:    %.2f ns                               ║",
                     avg_read_latency);
        end
        $display("╚════════════════════════════════════════════════════════════════╝");
        $display("");
        
        // Exit with appropriate code
        if (test_fail == 0) begin
            $display("✓ Testbench PASSED");
            $finish(0);
        end else begin
            $display("✗ Testbench FAILED");
            $finish(1);
        end
    end
    
    // ========================================================================
    // Timeout Guard
    // ========================================================================
    initial begin
        #((TEST_TIMEOUT_US) * 1000 * 1000);  // Convert µs to ns
        $display("\n[ERROR] TIMEOUT: Test exceeded %0d µs", TEST_TIMEOUT_US);
        $display("[ERROR] This may indicate a testbench or DUT hang.");
        $finish(2);
    end

endmodule

`default_nettype wire
