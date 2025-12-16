`ifndef WGT_BUFFER_V
`define WGT_BUFFER_V
`default_nettype none
// =============================================================================
// wgt_buffer.sv — Double-Buffered Weight SRAM for Systolic Array
// =============================================================================
//
// OVERVIEW
// ========
// This module provides ping-pong buffered storage for weight vectors.
// Structurally identical to act_buffer but optimized for weight access patterns.
// While the DMA fills one bank with next layer's weights, the systolic array
// reads from the other bank for current computation.
//
// WEIGHT VS ACTIVATION BUFFERS
// ============================
// ┌─────────────────────────────────────────────────────────────────────┐
// │ Buffer Type │ Dimension │ Access Pattern │ Typical Size            │
// ├─────────────────────────────────────────────────────────────────────┤
// │ act_buffer  │ TM (rows) │ Streaming K    │ 14×128 = 1,792 bytes    │
// │ wgt_buffer  │ TN (cols) │ Streaming K    │ 14×128 = 1,792 bytes    │
// └─────────────────────────────────────────────────────────────────────┘
//
// For weight-stationary dataflow:
//   - Weights are loaded once per tile, then held in PEs
//   - This buffer feeds one column of weights per cycle during load phase
//   - TN matches N_COLS (systolic array column count)
//
// MEMORY ORGANIZATION
// ===================
// Each bank stores a tile of weights:
//   - Width: TN × 8 bits (e.g., 14 INT8 values = 112 bits per column)
//   - Depth: 2^ADDR_WIDTH entries (e.g., 128 rows for ADDR_WIDTH=7)
//   - Total per bank: TN × 8 × 2^ADDR_WIDTH bits
//
// For dense layers, the full weight matrix is tiled:
//   Matrix [M×K] × [K×N] tiled as [TM×TK] × [TK×TN]
//   where TM=TN=14 and TK varies by layer
//
// CLOCK GATING
// ============
// Identical to act_buffer - saves ~85 mW when idle.
// Gate condition: !we && !rd_en
//
// TIMING
// ======
// - Write: Data captured on rising edge when we=1
// - Read: 1-cycle latency (registered output)
//         Cycle 0: rd_en=1, k_idx valid
//         Cycle 1: b_vec valid with data from mem[k_idx]
//
// MAGIC NUMBERS
// =============
// TN = 128:         Default column count (legacy, usually overridden to 14)
// ADDR_WIDTH = 7:   128 entries per bank
// 85 mW:            Clock gating power savings
//
// Requirements Trace:
//   REQ-ACCEL-BUF-01: Support double-buffered (ping/pong) operation.
//   REQ-ACCEL-BUF-02: Provide TN-wide INT8 vector output, 1-cycle latency.
//   REQ-ACCEL-BUF-03: Parameterizable depth for CNN tile size.
//
// =============================================================================

module wgt_buffer #(
    // =========================================================================
    // PARAMETER: TN - Weight Vector Width
    // =========================================================================
    // Number of INT8 weights read/written per cycle.
    // Should match systolic array column count (N_COLS parameter).
    // Default 128 is legacy; typically overridden to 14 for our 14×14 array.
    parameter TN = 128,
    
    // =========================================================================
    // PARAMETER: ADDR_WIDTH - Address Bit Width
    // =========================================================================
    // Determines tile depth: 2^ADDR_WIDTH entries per bank.
    // Must match act_buffer's ADDR_WIDTH for consistent tiling.
    parameter ADDR_WIDTH = 7,
    
    // =========================================================================
    // PARAMETER: ENABLE_CLOCK_GATING - Power Optimization
    // =========================================================================
    // 1 = Gate clock when buffer is idle
    // 0 = Always-on clock (for debug)
    parameter ENABLE_CLOCK_GATING = 1
)(
    // =========================================================================
    // SYSTEM INTERFACE
    // =========================================================================
    input  wire                  clk,    // System clock
    input  wire                  rst_n,  // Active-low async reset
    
    // =========================================================================
    // HOST WRITE PORT (from DMA)
    // =========================================================================
    /**
     * Write interface for loading weights from DDR via DMA.
     * For sparse operation, weights come from bsr_dma's wgt_wdata output.
     */
    input  wire                  we,          // Write enable
    input  wire [ADDR_WIDTH-1:0] waddr,       // Write address (row index in tile)
    input  wire [TN*8-1:0]       wdata,       // Write data (TN INT8 values)
    input  wire                  bank_sel_wr, // Bank select (0=Bank0, 1=Bank1)
    
    // =========================================================================
    // ARRAY READ PORT (to systolic array)
    // =========================================================================
    /**
     * Read interface for loading weights into PEs.
     * In weight-stationary mode, this is used during the "load" phase.
     * Once weights are in PEs, this port goes idle (clock gated).
     */
    input  wire                  rd_en,       // Read enable
    input  wire [ADDR_WIDTH-1:0] k_idx,       // Read address (K-dimension index)
    input  wire                  bank_sel_rd, // Bank select (0=Bank0, 1=Bank1)
    output reg  [TN*8-1:0]       b_vec        // Output weight vector (1-cycle latency)
);

    // =========================================================================
    // CLOCK GATING LOGIC
    // =========================================================================
    // Same implementation as act_buffer - see that module for detailed comments.
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

    // =========================================================================
    // MEMORY BANKS
    // =========================================================================
    // Two identical SRAM banks for ping-pong operation.
    // Synthesis infers BRAM on Xilinx.
    reg [TN*8-1:0] mem0 [0:(1<<ADDR_WIDTH)-1];
    reg [TN*8-1:0] mem1 [0:(1<<ADDR_WIDTH)-1];

    // =========================================================================
    // ASSERTIONS (Simulation-Only)
    // =========================================================================
    initial begin
        assert (TN > 0 && TN <= 1024)
            else $fatal("wgt_buffer: TN=%0d out of range (1-1024)", TN);
        assert (ADDR_WIDTH > 0 && ADDR_WIDTH <= 12)
            else $fatal("wgt_buffer: ADDR_WIDTH=%0d out of range (1-12)", ADDR_WIDTH);
    end

    always @(posedge clk) begin
        if (we) assert (waddr < (1<<ADDR_WIDTH))
            else $error("wgt_buffer: waddr %0d >= depth %0d", waddr, (1<<ADDR_WIDTH));
        if (rd_en) assert (k_idx < (1<<ADDR_WIDTH))
            else $error("wgt_buffer: k_idx %0d >= depth %0d", k_idx, (1<<ADDR_WIDTH));
    end

    always @(posedge clk) begin
        assert (bank_sel_wr == 1'b0 || bank_sel_wr == 1'b1)
            else $error("wgt_buffer: bank_sel_wr=%b invalid", bank_sel_wr);
        assert (bank_sel_rd == 1'b0 || bank_sel_rd == 1'b1)
            else $error("wgt_buffer: bank_sel_rd=%b invalid", bank_sel_rd);
    end

    // Coverage hooks (for UVM or functional coverage)
    // covergroup cg_wgt_write @(posedge clk);
    //   coverpoint waddr;
    //   coverpoint bank_sel_wr;
    // endgroup
    // cg_wgt_write cg = new();

    // =========================================================================
    // WRITE LOGIC (Host/DMA Side)
    // =========================================================================
    always @(posedge buf_gated_clk) begin
        if (we) begin
            if (bank_sel_wr == 1'b0)
                mem0[waddr] <= wdata;
            else
                mem1[waddr] <= wdata;
        end
    end

    // =========================================================================
    // READ LOGIC (Systolic Array Side)
    // =========================================================================
    // 1-cycle latency, same as act_buffer
    reg [TN*8-1:0] read_data;
    
    always @(posedge buf_gated_clk) begin
        if (rd_en) begin
            if (bank_sel_rd == 1'b0)
                read_data <= mem0[k_idx];
            else
                read_data <= mem1[k_idx];
        end
    end

    // Output register with reset
    always @(posedge buf_gated_clk or negedge rst_n) begin
        if (!rst_n)
            b_vec <= {TN*8{1'b0}};
        else
            b_vec <= read_data;
    end

endmodule
`default_nettype wire
`endif
// -----------------------------------------------------------------------------
// Example instantiation for 128-wide weights, 128-depth tile:
//
// wgt_buffer #(
//     .TN(128),
//     .ADDR_WIDTH(7)
// ) u_wgt_buffer (
//     .clk(clk),
//     .rst_n(rst_n),
//     .we(host_we),
//     .waddr(host_waddr),
//     .wdata(host_wdata),
//     .bank_sel_wr(host_bank_sel),
//     .rd_en(array_rd_en),
//     .k_idx(array_k_idx),
//     .bank_sel_rd(array_bank_sel),
//     .b_vec(b_vec_out)
// );
// -----------------------------------------------------------------------------
