`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// tb_systolic_array.v - Verilog-2001 integration testbench for systolic_array
// Streams K slices for Tk in {3,5,8,64} into a 2x2 array and writes CSVs.
// CSV outputs (under tb/integration/): A_inputs.csv, B_inputs.csv, C_results.csv
// -----------------------------------------------------------------------------
module tb_systolic_array;
    // Parameters
    parameter N_ROWS = 2;
    parameter N_COLS = 2;

    reg clk;
    reg rst_n;
    reg en;
    reg clr;

    reg  [N_ROWS*8-1:0] a_in_flat;
    reg  [N_COLS*8-1:0] b_in_flat;
    wire [N_ROWS*N_COLS*32-1:0] c_out_flat;

    integer fhA, fhB, fhC;

    // Instantiate DUT
    systolic_array #( .N_ROWS(N_ROWS), .N_COLS(N_COLS), .PIPE(1), .SAT(0) ) dut (
        .clk(clk), .rst_n(rst_n), .en(en), .clr(clr),
        .a_in_flat(a_in_flat), .b_in_flat(b_in_flat),
        .c_out_flat(c_out_flat)
    );

    // Clock
    initial clk = 0;
    always #5 clk = ~clk; // 100 MHz

    integer Tk_cases [0:3];
    initial begin
        Tk_cases[0] = 3; Tk_cases[1] = 5; Tk_cases[2] = 8; Tk_cases[3] = 64;
    end

    integer cycle;
    integer idx;

    // Helper: extract acc value for (r,c)
    function [31:0] get_c;
        input integer r; input integer c;
        begin
            get_c = c_out_flat[(r*N_COLS + c)*32 +: 32];
        end
    endfunction

    // Random INT8 helper
    function signed [7:0] rand_int8;
        input integer base;
        begin
            rand_int8 = $signed($urandom(base) & 8'hFF);
        end
    endfunction

    integer r,c,k;

    initial begin
        fhA = $fopen("tb/integration/A_inputs.csv","w");
        fhB = $fopen("tb/integration/B_inputs.csv","w");
        fhC = $fopen("tb/integration/C_results.csv","w");
        if (!fhA || !fhB || !fhC) begin
            $display("ERROR: cannot open one or more CSV files");
            $finish;
        end
        $fwrite(fhA, "Tk,r,k,val\n");
        $fwrite(fhB, "Tk,k,c,val\n");
        $fwrite(fhC, "Tk,r,c,val\n");

        rst_n = 0; en = 0; clr = 0; a_in_flat = 0; b_in_flat = 0; cycle = 0;
        repeat(5) @(posedge clk);
        rst_n = 1; repeat(5) @(posedge clk);

        for (idx = 0; idx < 4; idx = idx + 1) begin
            integer Tk;
            Tk = Tk_cases[idx];
            // Clear accumulators
            clr = 1; en = 0; @(posedge clk); clr = 0;

            // Feed K slices
            for (k = 0; k < Tk; k = k + 1) begin
                // Build a_in_flat and b_in_flat
                for (r = 0; r < N_ROWS; r = r + 1) begin
                    // pack each row's byte at position r
                    a_in_flat[r*8 +: 8] = rand_int8(32'hA000 + (Tk<<8) + (r<<4) + k);
                    $fwrite(fhA, "%0d,%0d,%0d,%0d\n", Tk, r, k, $signed(a_in_flat[r*8 +: 8]));
                end
                for (c = 0; c < N_COLS; c = c + 1) begin
                    b_in_flat[c*8 +: 8] = rand_int8(32'hB000 + (Tk<<8) + (c<<4) + k);
                    $fwrite(fhB, "%0d,%0d,%0d,%0d\n", Tk, k, c, $signed(b_in_flat[c*8 +: 8]));
                end
                en = 1; @(posedge clk);
            end
            // stop feeding
            en = 0; a_in_flat = 0; b_in_flat = 0;
            // Drain cycles (N_ROWS + N_COLS conservative)
            repeat (N_ROWS + N_COLS) @(posedge clk);

            // Capture outputs
            for (r = 0; r < N_ROWS; r = r + 1) begin
                for (c = 0; c < N_COLS; c = c + 1) begin
                    $fwrite(fhC, "%0d,%0d,%0d,%0d\n", Tk, r, c, $signed(get_c(r,c)));
                end
            end
            // gap
            repeat(5) @(posedge clk);
        end

        $fclose(fhA); $fclose(fhB); $fclose(fhC);
        #50;
        $display("tb_systolic_array: COMPLETE");
        $finish;
    end
endmodule
`default_nettype wire
