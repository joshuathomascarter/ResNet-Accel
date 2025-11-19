`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// tb_mac8.v - Verilog-2001 unit testbench for mac8
// Generates random and edge-case vectors; writes CSV results for Python golden.
// -----------------------------------------------------------------------------
// Output CSV: tb/unit/mac8_results.csv
// Columns: cycle,a,b,en,clr,acc,sat
// -----------------------------------------------------------------------------
module tb_mac8;
    // Parameters for test
    integer SEEDS [0:4];
    integer seed_index;
    integer cycle;
    integer fh;

    // DUT signals
    reg clk;
    reg rst_n;
    reg signed [7:0] a;
    reg signed [7:0] b;
    reg clr;
    reg en;
    wire signed [31:0] acc;
    wire sat_flag;

    // Instantiate DUT (test saturation enabled and disabled in two passes?)
    // Simpler: single instance SAT=1 to exercise overflow signaling.
    mac8 #( .SAT(1) ) dut (
        .clk(clk), .rst_n(rst_n),
        .a(a), .b(b),
        .clr(clr), .en(en),
        .acc(acc), .sat_flag(sat_flag)
    );

    // Clock
    initial clk = 0;
    always #5 clk = ~clk; // 100MHz

    // Initialize seeds array
    initial begin
        SEEDS[0] = 32'h1001;
        SEEDS[1] = 32'h2002;
        SEEDS[2] = 32'h3003;
        SEEDS[3] = 32'h4004;
        SEEDS[4] = 32'h5005;
    end

    // Test procedure
    initial begin
        fh = $fopen("tb/unit/mac8_results.csv","w");
        if (fh == 0) begin
            $display("ERROR: cannot open output CSV");
            $finish;
        end
        $fwrite(fh, "cycle,a,b,en,clr,acc,sat\n");

        rst_n = 0; a = 0; b = 0; clr = 0; en = 0; cycle = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        // Loop over seeds
        for (seed_index = 0; seed_index < 5; seed_index = seed_index + 1) begin
            // Clear before each seed run
            clr = 1; en = 0; @(posedge clk); cycle = cycle + 1; write_line(); clr = 0;
            @(posedge clk); cycle = cycle + 1; write_line();

            // Edge sequence: 64 cycles of max positive to verify worst-case accumulation
            integer k;
            for (k = 0; k < 64; k = k + 1) begin
                a = 8'sd127; b = 8'sd127; en = 1; clr = 0; @(posedge clk); cycle = cycle + 1; write_line();
            end
            en = 0; @(posedge clk); cycle = cycle + 1; write_line();

            // Random vectors
            integer rv;
            for (rv = 0; rv < 256; rv = rv + 1) begin
                a = $signed($urandom(SEEDS[seed_index] + rv) & 8'hFF);
                b = $signed($urandom(SEEDS[seed_index] + rv + 32'h55AA) & 8'hFF);
                en = 1; clr = 0; @(posedge clk); cycle = cycle + 1; write_line();
            end
            en = 0; @(posedge clk); cycle = cycle + 1; write_line();

            // Additional edge cases
            a = -128; b = -1; en = 1; @(posedge clk); cycle = cycle + 1; write_line();
            a = -128; b = 1;  en = 1; @(posedge clk); cycle = cycle + 1; write_line();
            a = 1;    b = -128; en = 1; @(posedge clk); cycle = cycle + 1; write_line();
            en = 0; @(posedge clk); cycle = cycle + 1; write_line();
        end

        $fclose(fh);
        #50;
        $display("tb_mac8: COMPLETE");
        $finish;
    end

    task write_line; begin
        $fwrite(fh, "%0d,%0d,%0d,%0d,%0d,%0d,%0d\n", cycle, a, b, en, clr, acc, sat_flag);
    end endtask

endmodule
`default_nettype wire
