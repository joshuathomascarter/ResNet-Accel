`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// tb_pe.v - Verilog-2001 unit testbench for pe
// Verifies pipeline skew (PIPE=1), accumulation via mac8, clear behavior.
// Writes CSV: tb/unit/pe_results.csv
// Columns: cycle,a_in,b_in,en,clr,a_out,b_out,acc
// -----------------------------------------------------------------------------
module tb_pe;
    reg clk;
    reg rst_n;
    reg signed [7:0] a_in;
    reg signed [7:0] b_in;
    reg en;
    reg clr;
    wire signed [7:0] a_out;
    wire signed [7:0] b_out;
    wire signed [31:0] acc;

    integer fh;
    integer cycle;

    // DUT (PIPE=1, SAT=0)
    pe #( .PIPE(1), .SAT(0) ) dut (
        .clk(clk), .rst_n(rst_n),
        .a_in(a_in), .b_in(b_in),
        .en(en), .clr(clr),
        .a_out(a_out), .b_out(b_out),
        .acc(acc)
    );

    // Clock
    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        fh = $fopen("tb/unit/pe_results.csv","w");
        if (fh == 0) begin
            $display("ERROR: cannot open pe_results.csv");
            $finish;
        end
        $fwrite(fh, "cycle,a_in,b_in,en,clr,a_out,b_out,acc\n");

        rst_n = 0; a_in = 0; b_in = 0; en = 0; clr = 0; cycle = 0;
        repeat(5) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);

        // Clear
        clr = 1; en = 0; @(posedge clk); cycle = cycle + 1; write_line(); clr = 0;
        @(posedge clk); cycle = cycle + 1; write_line();

        // 3-cycle microstream (2,3),(4,5),(-3,7)
        apply_pair(8'sd2, 8'sd3, 1);
        apply_pair(8'sd4, 8'sd5, 1);
        apply_pair(-8'sd3, 8'sd7, 1);
        // idle
        apply_pair(8'sd0, 8'sd0, 0);
        // clear
        clr = 1; en = 0; @(posedge clk); cycle = cycle + 1; write_line(); clr = 0;
        @(posedge clk); cycle = cycle + 1; write_line();

        // Random streaming (16 cycles)
        integer i;
        for (i = 0; i < 16; i = i + 1) begin
            apply_pair($signed($urandom(32'hCAFE+i) & 8'hFF), $signed($urandom(32'hBABE+i) & 8'hFF), 1);
        end

        // Finish
        $fclose(fh);
        #50;
        $display("tb_pe: COMPLETE");
        $finish;
    end

    task apply_pair;
        input signed [7:0] va;
        input signed [7:0] vb;
        input integer do_en;
        begin
            a_in = va; b_in = vb; en = do_en; clr = 0;
            @(posedge clk); cycle = cycle + 1; write_line();
        end
    endtask

    task write_line; begin
        $fwrite(fh, "%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d\n", cycle, a_in, b_in, en, clr, a_out, b_out, acc);
    end endtask

endmodule
`default_nettype wire
