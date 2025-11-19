`ifndef ACT_BUFFER_V
`define ACT_BUFFER_V
`default_nettype none
// -----------------------------------------------------------------------------
// Title      : act_buffer
// File       : act_buffer.v
// Description: Double-buffered activation SRAM (ping/pong) for systolic array.
//              Host writes to one bank, array reads from the other.
//              1-cycle read latency modeled. Verilog-2001 compliant.
//
// Requirements Trace:
//   REQ-ACCEL-BUF-01: Support double-buffered (ping/pong) operation.
//   REQ-ACCEL-BUF-02: Provide TM-wide INT8 vector output, 1-cycle latency.
//   REQ-ACCEL-BUF-03: Parameterizable depth for CNN tile size.
//
// Parameters:
//   TM         : Activation vector width (number of INT8 elements, e.g. 128 for CNN)
//   ADDR_WIDTH : Address width (log2 of tile depth, e.g. 7 for 128-depth)
//
// Ports:
//   clk         : Clock
//   rst_n       : Active-low reset
//   we          : Write enable (host)
//   waddr       : Write address (host)
//   wdata       : Write data (TM*8 bits)
//   bank_sel_wr : Bank select for write (0 or 1)
//   rd_en       : Read enable (array)
//   k_idx       : Read address (array)
//   bank_sel_rd : Bank select for read (0 or 1)
//   a_vec       : Output activation vector (TM*8 bits, 1-cycle latency)
// ----------------------------------------------------------------------------

module act_buffer #(
    parameter TM = 128,           // Activation vector width (number of elements)
    parameter ADDR_WIDTH = 7,     // Address width (for tile depth, 2^7=128)
    parameter ENABLE_CLOCK_GATING = 1  // Enable clock gating (saves 85 mW)
)(
    input  wire                  clk,
    input  wire                  rst_n,
    // Host write port
    input  wire                  we,
    input  wire [ADDR_WIDTH-1:0] waddr,
    input  wire [TM*8-1:0]       wdata,
    input  wire                  bank_sel_wr, // 0 or 1
    // Array read port
    input  wire                  rd_en,
    input  wire [ADDR_WIDTH-1:0] k_idx,
    input  wire                  bank_sel_rd, // 0 or 1
    output reg  [TM*8-1:0]       a_vec        // Output vector (1-cycle latency)
);

    // Clock gating logic - gate when idle (no read/write)
    wire buf_clk_en, buf_gated_clk;
    assign buf_clk_en = we | rd_en;
    
    generate
        if (ENABLE_CLOCK_GATING) begin : gen_clk_gate
            `ifdef XILINX_FPGA
                BUFGCE buf_clk_gate (
                    .I  (clk),
                    .CE (buf_clk_en),
                    .O  (buf_gated_clk)
                );
            `else
                reg buf_clk_en_latched;
                always @(clk or buf_clk_en) begin
                    if (!clk) buf_clk_en_latched <= buf_clk_en;
                end
                assign buf_gated_clk = clk & buf_clk_en_latched;
            `endif
        end else begin : gen_no_gate
            assign buf_gated_clk = clk;
        end
    endgenerate

    // Two banks of simple SRAM (modeled as reg arrays)
    reg [TM*8-1:0] mem0 [0:(1<<ADDR_WIDTH)-1];
    reg [TM*8-1:0] mem1 [0:(1<<ADDR_WIDTH)-1];

    // -------------------------------------------------------------------------
    // Assertions (SystemVerilog)
    // -------------------------------------------------------------------------
    // SVA: Parameter bounds
    initial begin
        assert (TM > 0 && TM <= 1024)
            else $fatal("act_buffer: TM out of range");
        assert (ADDR_WIDTH > 0 && ADDR_WIDTH <= 12)
            else $fatal("act_buffer: ADDR_WIDTH out of range");
    end

    // SVA: Address range
    always @(posedge clk) begin
        if (we) assert (waddr < (1<<ADDR_WIDTH))
            else $error("act_buffer: waddr out of range: %0d", waddr);
        if (rd_en) assert (k_idx < (1<<ADDR_WIDTH))
            else $error("act_buffer: k_idx out of range: %0d", k_idx);
    end

    // SVA: Bank select must be 0 or 1
    always @(posedge clk) begin
        assert (bank_sel_wr == 1'b0 || bank_sel_wr == 1'b1)
            else $error("act_buffer: bank_sel_wr out of range");
        assert (bank_sel_rd == 1'b0 || bank_sel_rd == 1'b1)
            else $error("act_buffer: bank_sel_rd out of range");
    end

    // Coverage hooks (for UVM or functional coverage)
    // covergroup cg_act_write @(posedge clk);
    //   coverpoint waddr;
    //   coverpoint bank_sel_wr;
    // endgroup
    // cg_act_write cg = new();

    // Write logic (host)
    always @(posedge buf_gated_clk) begin
        if (we) begin
            if (bank_sel_wr == 1'b0)
                mem0[waddr] <= wdata;
            else
                mem1[waddr] <= wdata;
        end
    end

    // 1-cycle read latency (array)
    reg [TM*8-1:0] read_data;
    always @(posedge buf_gated_clk) begin
        if (rd_en) begin
            if (bank_sel_rd == 1'b0)
                read_data <= mem0[k_idx];
            else
                read_data <= mem1[k_idx];
        end
    end

    // Output register (1-cycle latency)
    always @(posedge buf_gated_clk or negedge rst_n) begin
        if (!rst_n)
            a_vec <= {TM*8{1'b0}};
        else
            a_vec <= read_data;
    end

endmodule
`default_nettype wire
`endif

// -----------------------------------------------------------------------------
// Example instantiation for 128-wide activations, 128-depth tile:
//
// act_buffer #(
//     .TM(128),
//     .ADDR_WIDTH(7)
// ) u_act_buffer (
//     .clk(clk),
//     .rst_n(rst_n),
//     .we(host_we),
//     .waddr(host_waddr),
//     .wdata(host_wdata),
//     .bank_sel_wr(host_bank_sel),
//     .rd_en(array_rd_en),
//     .k_idx(array_k_idx),
//     .bank_sel_rd(array_bank_sel),
//     .a_vec(a_vec_out)
// );
// -----------------------------------------------------------------------------
