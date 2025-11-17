// accel_top.v
// Complete UART-based ACCEL-v1 accelerator top-level
// Full packet protocol with CSR, buffer management, and computation control

`default_nettype none

module accel_top #(
    parameter N_ROWS = 2,
    parameter N_COLS = 2,
    parameter TM = 8,
    parameter TN = 8, 
    parameter TK = 8,
    parameter CLK_HZ = 50_000_000,
    parameter BAUD = 115_200,
    parameter ADDR_WIDTH = 6
)(
    input  wire clk,
    input  wire rst_n,
    
    // UART interface
    input  wire uart_rx,
    output wire uart_tx,
    
    // AXI4-Lite Host Interface (Phase 4)
    input  wire [31:0] s_axi_awaddr,
    input  wire [1:0]  s_axi_awburst,
    input  wire [7:0]  s_axi_awlen,
    input  wire [2:0]  s_axi_awsize,
    input  wire        s_axi_awvalid,
    output wire        s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wlast,
    input  wire        s_axi_wvalid,
    output wire        s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output wire        s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [31:0] s_axi_araddr,
    input  wire [1:0]  s_axi_arburst,
    input  wire [7:0]  s_axi_arlen,
    input  wire [2:0]  s_axi_arsize,
    input  wire        s_axi_arvalid,
    output wire        s_axi_arready,
    output wire [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output wire        s_axi_rlast,
    output wire        s_axi_rvalid,
    input  wire        s_axi_rready,
    
    // Status outputs
    output wire busy,
    output wire done_pulse,
    output wire error
);

    // ========================================================================
    // Internal Wiring
    // ========================================================================
    
    // CSR interface
    wire csr_wen, csr_ren;
    wire [7:0] csr_addr;
    wire [31:0] csr_wdata, csr_rdata;
    wire start_pulse, abort_pulse, irq_en;
    
    // CSR configuration outputs
    wire [31:0] M, N, K;
    wire [31:0] Tm, Tn, Tk;
    wire [31:0] m_idx, n_idx, k_idx;
    wire bank_sel_wr_A, bank_sel_wr_B;
    wire bank_sel_rd_A, bank_sel_rd_B;
    wire [31:0] Sa_bits, Sw_bits;
    wire [31:0] uart_len_max;
    wire uart_crc_en;
    
    // Systolic array signals
    wire array_en, array_clr, load_weight;  // Added load_weight for row-stationary
    wire [(N_ROWS*8)-1:0] a_in_flat;
    wire [(N_COLS*8)-1:0] b_in_flat;
    wire [(N_ROWS*N_COLS*32)-1:0] c_out_flat;
    
    // Performance monitor signals
    wire [31:0] perf_total_cycles;
    wire [31:0] perf_active_cycles;
    wire [31:0] perf_idle_cycles;
    wire perf_measurement_done;
    
    // Buffer signals
    wire act_we, wgt_we, out_we;
    wire [TM*8-1:0] act_wdata;
    wire [TN*8-1:0] wgt_wdata;
    wire [N_ROWS*N_COLS*32-1:0] out_wdata;
    wire [TM*8-1:0] a_vec;
    wire [TN*8-1:0] b_vec;
    wire [ADDR_WIDTH-1:0] act_waddr, wgt_waddr, out_waddr;
    wire [ADDR_WIDTH-1:0] act_k_idx, wgt_k_idx;
    wire act_rd_en, wgt_rd_en;
    
    // UART physical layer
    wire [7:0] uart_rx_data, uart_tx_data;
    wire uart_rx_valid, uart_tx_ready, uart_tx_valid;
    wire uart_rx_frm_err, uart_rx_par_err;
    
    // Scheduler signals
    wire sched_busy, sched_done_tile;
    wire [9:0] sched_m_tile, sched_n_tile;
    wire [11:0] sched_k_tile;
    wire [31:0] cycles_tile, stall_cycles;
    wire [2:0] k_idx_sched;
    wire [TM-1:0] en_mask_row;
    wire [TN-1:0] en_mask_col;
    
    // Status outputs
    assign busy = sched_busy;
    assign done_pulse = sched_done_tile;
    assign error = uart_rx_frm_err | uart_rx_par_err;

    // ========================================================================
    // UART Packet Protocol Handler
    // ========================================================================
    // Packet format: [CMD][ADDR_L][ADDR_H][DATA_0][DATA_1][DATA_2][DATA_3]
    // CMD byte encoding:
    //   [7:4] = command type: 0x0=CSR_WR, 0x1=CSR_RD, 0x2=BUF_WR_A, 0x3=BUF_WR_B, 
    //           0x4=BUF_RD_OUT, 0x5=START, 0x6=ABORT
    //   [3:0] = sub-command or flags
    
    localparam [3:0] CMD_CSR_WR     = 4'h0,
                     CMD_CSR_RD     = 4'h1,
                     CMD_BUF_WR_A   = 4'h2,
                     CMD_BUF_WR_B   = 4'h3,
                     CMD_BUF_RD_OUT = 4'h4,
                     CMD_START      = 4'h5,
                     CMD_ABORT      = 4'h6,
                     CMD_STATUS     = 4'h7;
    
    // UART packet receiver state machine
    localparam [2:0] RX_IDLE   = 3'd0,
                     RX_CMD    = 3'd1,
                     RX_ADDR_L = 3'd2,
                     RX_ADDR_H = 3'd3,
                     RX_DATA0  = 3'd4,
                     RX_DATA1  = 3'd5,
                     RX_DATA2  = 3'd6,
                     RX_DATA3  = 3'd7;
    
    reg [2:0] rx_state;
    reg [7:0] rx_cmd;
    reg [15:0] rx_addr;
    reg [31:0] rx_data;
    reg rx_valid;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            rx_state <= RX_IDLE;
            rx_cmd <= 8'h0;
            rx_addr <= 16'h0;
            rx_data <= 32'h0;
            rx_valid <= 1'b0;
        end else begin
            rx_valid <= 1'b0; // Pulse output
            
            if (uart_rx_valid) begin
                case (rx_state)
                    RX_IDLE: begin
                        rx_cmd <= uart_rx_data;
                        rx_state <= RX_CMD;
                    end
                    RX_CMD: begin
                        rx_addr[7:0] <= uart_rx_data;
                        rx_state <= RX_ADDR_L;
                    end
                    RX_ADDR_L: begin
                        rx_addr[15:8] <= uart_rx_data;
                        rx_state <= RX_ADDR_H;
                    end
                    RX_ADDR_H: begin
                        rx_data[7:0] <= uart_rx_data;
                        rx_state <= RX_DATA0;
                    end
                    RX_DATA0: begin
                        rx_data[15:8] <= uart_rx_data;
                        rx_state <= RX_DATA1;
                    end
                    RX_DATA1: begin
                        rx_data[23:16] <= uart_rx_data;
                        rx_state <= RX_DATA2;
                    end
                    RX_DATA2: begin
                        rx_data[31:24] <= uart_rx_data;
                        rx_state <= RX_DATA3;
                    end
                    RX_DATA3: begin
                        rx_valid <= 1'b1;
                        rx_state <= RX_IDLE;
                    end
                    default: rx_state <= RX_IDLE;
                endcase
            end
        end
    end
    
    // Command decoder
    wire [3:0] cmd_type = rx_cmd[7:4];
    wire cmd_is_csr_wr     = rx_valid && (cmd_type == CMD_CSR_WR);
    wire cmd_is_csr_rd     = rx_valid && (cmd_type == CMD_CSR_RD);
    wire cmd_is_buf_wr_a   = rx_valid && (cmd_type == CMD_BUF_WR_A);
    wire cmd_is_buf_wr_b   = rx_valid && (cmd_type == CMD_BUF_WR_B);
    wire cmd_is_buf_rd_out = rx_valid && (cmd_type == CMD_BUF_RD_OUT);
    wire cmd_is_start      = rx_valid && (cmd_type == CMD_START);
    wire cmd_is_abort      = rx_valid && (cmd_type == CMD_ABORT);
    wire cmd_is_status     = rx_valid && (cmd_type == CMD_STATUS);
    
    // CSR interface
    assign csr_wen = cmd_is_csr_wr;
    assign csr_ren = cmd_is_csr_rd;
    assign csr_addr = rx_addr[7:0];
    assign csr_wdata = rx_data;
    
    // Buffer write control
    reg [TM*8-1:0] act_wdata_r;
    reg [TN*8-1:0] wgt_wdata_r;
    reg [ADDR_WIDTH-1:0] act_waddr_r, wgt_waddr_r;
    reg act_we_r, wgt_we_r;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            act_we_r <= 1'b0;
            wgt_we_r <= 1'b0;
            act_waddr_r <= {ADDR_WIDTH{1'b0}};
            wgt_waddr_r <= {ADDR_WIDTH{1'b0}};
            act_wdata_r <= {TM*8{1'b0}};
            wgt_wdata_r <= {TN*8{1'b0}};
        end else begin
            // Default: disable writes
            act_we_r <= 1'b0;
            wgt_we_r <= 1'b0;
            
            if (cmd_is_buf_wr_a) begin
                act_we_r <= 1'b1;
                act_waddr_r <= rx_addr[ADDR_WIDTH-1:0];
                act_wdata_r <= rx_data[TM*8-1:0];
            end
            
            if (cmd_is_buf_wr_b) begin
                wgt_we_r <= 1'b1;
                wgt_waddr_r <= rx_addr[ADDR_WIDTH-1:0];
                wgt_wdata_r <= rx_data[TN*8-1:0];
            end
        end
    end
    
    assign act_we = act_we_r;
    assign wgt_we = wgt_we_r;
    assign act_waddr = act_waddr_r;
    assign wgt_waddr = wgt_waddr_r;
    assign act_wdata = act_wdata_r;
    assign wgt_wdata = wgt_wdata_r;
    
    // UART transmit response handler
    localparam [1:0] TX_IDLE    = 2'd0,
                     TX_HEADER  = 2'd1,
                     TX_DATA    = 2'd2,
                     TX_WAIT    = 2'd3;
    
    reg [1:0] tx_state;
    reg [2:0] tx_byte_count;
    reg [31:0] tx_data_reg;
    reg [7:0] tx_cmd_reg;
    reg tx_active;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            tx_state <= TX_IDLE;
            tx_byte_count <= 3'd0;
            tx_data_reg <= 32'h0;
            tx_cmd_reg <= 8'h0;
            tx_active <= 1'b0;
        end else begin
            case (tx_state)
                TX_IDLE: begin
                    if (cmd_is_csr_rd || cmd_is_status) begin
                        tx_cmd_reg <= rx_cmd;
                        tx_data_reg <= (cmd_is_status) ? {30'h0, error, busy} : csr_rdata;
                        tx_byte_count <= 3'd0;
                        tx_state <= TX_HEADER;
                        tx_active <= 1'b1;
                    end else if (cmd_is_buf_rd_out) begin
                        tx_cmd_reg <= rx_cmd;
                        tx_data_reg <= c_out_flat[31:0]; // Return first word
                        tx_byte_count <= 3'd0;
                        tx_state <= TX_HEADER;
                        tx_active <= 1'b1;
                    end
                end
                
                TX_HEADER: begin
                    if (uart_tx_ready) begin
                        tx_state <= TX_DATA;
                    end
                end
                
                TX_DATA: begin
                    if (uart_tx_ready) begin
                        tx_byte_count <= tx_byte_count + 1'b1;
                        if (tx_byte_count == 3'd3) begin
                            tx_state <= TX_WAIT;
                            tx_active <= 1'b0;
                        end
                    end
                end
                
                TX_WAIT: begin
                    tx_state <= TX_IDLE;
                end
                
                default: tx_state <= TX_IDLE;
            endcase
        end
    end
    
    assign uart_tx_valid = tx_active && (tx_state == TX_HEADER || tx_state == TX_DATA);
    assign uart_tx_data = (tx_state == TX_HEADER) ? tx_cmd_reg :
                         (tx_byte_count == 3'd0) ? tx_data_reg[7:0] :
                         (tx_byte_count == 3'd1) ? tx_data_reg[15:8] :
                         (tx_byte_count == 3'd2) ? tx_data_reg[23:16] :
                         tx_data_reg[31:24];

    // ========================================================================
    // Hardware Module Instantiations
    // ========================================================================


    // CSR (Control/Status Register) module
    csr #(
        .ADDR_W(8)
    ) csr_inst (
        .clk(clk),
        .rst_n(rst_n),
        .csr_wen(csr_wen),
        .csr_ren(csr_ren),
        .csr_addr(csr_addr),
        .csr_wdata(csr_wdata),
        .csr_rdata(csr_rdata),
        .core_busy(sched_busy),
        .core_done_tile_pulse(sched_done_tile),
        .core_bank_sel_rd_A(bank_sel_rd_A),
        .core_bank_sel_rd_B(bank_sel_rd_B),
        .rx_crc_error(uart_rx_par_err),
        .rx_illegal_cmd(uart_rx_frm_err),
        .start_pulse(start_pulse),
        .abort_pulse(abort_pulse),
        .irq_en(irq_en),
        .M(M), .N(N), .K(K),
        .Tm(Tm), .Tn(Tn), .Tk(Tk),
        .m_idx(m_idx), .n_idx(n_idx), .k_idx(k_idx),
        .bank_sel_wr_A(bank_sel_wr_A), 
        .bank_sel_wr_B(bank_sel_wr_B),
        .bank_sel_rd_A(bank_sel_rd_A), 
        .bank_sel_rd_B(bank_sel_rd_B),
        .Sa_bits(Sa_bits), 
        .Sw_bits(Sw_bits),
        .uart_len_max(uart_len_max),
        .uart_crc_en(uart_crc_en),
        .perf_total_cycles(perf_total_cycles),
        .perf_active_cycles(perf_active_cycles),
        .perf_idle_cycles(perf_idle_cycles)
    );

    // ========================================================================
    // BSR DMA Engine (loads sparse matrix data via UART)
    // ========================================================================
    bsr_dma #(
        .DATA_WIDTH(8),
        .ADDR_WIDTH(16),
        .MAX_LAYERS(8),
        .MAX_BLOCKS(65536),
        .BLOCK_SIZE(64),
        .ROW_PTR_DEPTH(256),
        .COL_IDX_DEPTH(65536),
        .ENABLE_CRC(0)
    ) bsr_dma_inst (
        .clk(clk),
        .rst_n(rst_n),
        .uart_rx_data(uart_rx_data),
        .uart_rx_valid(uart_rx_valid),
        .uart_rx_ready(),  // Can be connected to flow control if needed
        .uart_tx_data(),   // Status responses (optional)
        .uart_tx_valid(),
        .uart_tx_ready(uart_tx_ready),
        .csr_addr(csr_addr),
        .csr_wen(csr_wen),
        .csr_wdata(csr_wdata),
        .csr_rdata(),  // Can mux into CSR read path if needed
        .row_ptr_we(dma_row_ptr_we),
        .row_ptr_waddr(dma_row_ptr_waddr),
        .row_ptr_wdata(dma_row_ptr_wdata),
        .col_idx_we(dma_col_idx_we),
        .col_idx_waddr(dma_col_idx_waddr),
        .col_idx_wdata(dma_col_idx_wdata),
        .block_we(dma_block_we),
        .block_waddr(dma_block_waddr),
        .block_wdata(dma_block_wdata),
        .dma_busy(dma_busy),
        .dma_done(dma_done),
        .dma_error(dma_error),
        .blocks_written(dma_blocks_written)
    );

    // ========================================================================
    // SPARSE ACCELERATION MODULES (BSR Format)
    // ========================================================================
    
    // DMA interface signals for BSR metadata loading
    wire dma_busy, dma_done, dma_error;
    wire [31:0] dma_blocks_written;
    
    // BSR Scheduler signals
    wire bsr_start, bsr_done, bsr_busy;
    wire [15:0] bsr_num_block_rows, bsr_num_block_cols;
    wire [31:0] bsr_total_blocks;
    wire bsr_layer_switch;
    wire [2:0] bsr_active_layer;
    wire bsr_layer_ready;
    
    // DMA write interfaces for BSR metadata BRAMs
    wire dma_row_ptr_we;
    wire [15:0] dma_row_ptr_waddr;
    wire [31:0] dma_row_ptr_wdata;
    
    wire dma_col_idx_we;
    wire [15:0] dma_col_idx_waddr;
    wire [15:0] dma_col_idx_wdata;
    
    wire dma_block_we;
    // Block write interface is now word-addressed and 32-bit wide
    wire [20:0] dma_block_waddr;  // word address: (byte address >> 2)
    wire [31:0] dma_block_wdata;
    
    // BSR Metadata BRAMs (row_ptr, col_idx)
    wire row_ptr_rd_en;
    wire [15:0] row_ptr_rd_addr;
    wire [31:0] row_ptr_rd_data;
    wire col_idx_rd_en;
    wire [31:0] col_idx_rd_addr;
    wire [15:0] col_idx_rd_data;
    
    // Block data BRAM (32-bit word-addressed for DMA)
    wire block_rd_en;
    wire [31:0] block_rd_addr; // word address from scheduler
    wire [31:0] block_rd_data; // 32-bit word
    
    // AXI-Lite CSR Interface
    wire axi_csr_wen, axi_csr_ren;
    wire [7:0] axi_csr_addr;
    wire [31:0] axi_csr_wdata, axi_csr_rdata;
    
    // AXI DMA Bridge
    wire [31:0] axi_dma_fifo_wdata;
    wire axi_dma_fifo_wen;
    wire axi_dma_fifo_full;
    wire [6:0] axi_dma_fifo_count;
    wire axi_error;
    
    // CSR Source Mux (UART vs AXI)
    wire [31:0] csr_rdata_axi, csr_rdata_uart;
    wire csr_ren_uart, csr_wen_uart, csr_ren_axi, csr_wen_axi;
    
    // Scheduler to Sparse Systolic interface
    wire systolic_sparse_valid, systolic_sparse_ready, systolic_sparse_done;
    wire [7:0] systolic_sparse_block [0:63]; // 8×8 block
    wire [15:0] systolic_sparse_block_row, systolic_sparse_block_col;
    
    // Sparse Systolic to Reorder Buffer
    wire reorder_in_valid, reorder_in_row_done;
    wire [15:0] reorder_in_block_col;
    wire [31:0] reorder_in_block_idx;
    wire reorder_out_valid, reorder_out_row_done;
    wire [15:0] reorder_out_block_col;
    wire [31:0] reorder_out_block_idx;
    
    // Activation input for sparse systolic (8 INT8 values)
    wire [7:0] sparse_act_data [0:7];
    wire sparse_act_valid;
    
    // Sparse systolic result (2×8 INT32 results for 2×2 PEs)
    wire sparse_result_valid;
    wire [31:0] sparse_result_data [0:1][0:7]; // 2 rows × 8 columns
    wire [15:0] sparse_result_block_row, sparse_result_block_col;
    wire sparse_result_ready;
    
    // Assign BSR config from CSR (add these CSR addresses in csr.v)
    assign bsr_num_block_rows = M[15:0] >> 3;  // M/8 for 8×8 blocks
    assign bsr_num_block_cols = N[15:0] >> 3;  // N/8
    assign bsr_total_blocks = K[31:0];          // Reuse K as total blocks
    assign bsr_start = start_pulse;
    assign bsr_layer_switch = 1'b0;  // Static for now, add CSR bit later
    assign bsr_active_layer = 3'd0;   // Layer 0 by default
    
    // Connect activations from act_buffer to sparse systolic
    // For now, direct wire - add proper scheduling later
    assign sparse_act_valid = act_rd_en;
    genvar act_i;
    generate
        for (act_i = 0; act_i < 8; act_i = act_i + 1) begin : gen_sparse_act
            assign sparse_act_data[act_i] = a_vec[act_i*8 +: 8];
        end
    endgenerate
    
    // BSR Scheduler (traverses sparse blocks)
    bsr_scheduler #(
        .BLOCK_H(8),
        .BLOCK_W(8),
        .BLOCK_SIZE(64),
        .MAX_BLOCK_ROWS(256),
        .MAX_BLOCKS(65536),
        .DATA_WIDTH(8)
    ) bsr_scheduler_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(bsr_start),
        .cfg_num_block_rows(bsr_num_block_rows),
        .cfg_num_block_cols(bsr_num_block_cols),
        .cfg_total_blocks(bsr_total_blocks),
        .cfg_layer_switch(bsr_layer_switch),
        .cfg_active_layer(bsr_active_layer),
        .cfg_layer_ready(bsr_layer_ready),
        .row_ptr_rd_en(row_ptr_rd_en),
        .row_ptr_rd_addr(row_ptr_rd_addr),
        .row_ptr_rd_data(row_ptr_rd_data),
        .col_idx_rd_en(col_idx_rd_en),
        .col_idx_rd_addr(col_idx_rd_addr),
        .col_idx_rd_data(col_idx_rd_data),
        .block_rd_en(block_rd_en),
        .block_rd_addr(block_rd_addr),
        .block_rd_data(block_rd_data),
        .systolic_valid(systolic_sparse_valid),
        .systolic_block(systolic_sparse_block),
        .systolic_block_row(systolic_sparse_block_row),
        .systolic_block_col(systolic_sparse_block_col),
        .systolic_ready(systolic_sparse_ready),
        .systolic_done(systolic_sparse_done),
        .done(bsr_done),
        .busy(bsr_busy),
        .blocks_processed()  // Monitor in perf module
    );
    
    // Sparse Systolic Array Wrapper (processes 8×8 blocks with 2×2 PEs)
    systolic_array_sparse #(
        .PE_ROWS(2),
        .PE_COLS(2),
        .DATA_WIDTH(8),
        .ACC_WIDTH(32),
        .BLOCK_H(8),
        .BLOCK_W(8)
    ) systolic_sparse_inst (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(systolic_sparse_valid),
        .block_data(systolic_sparse_block),
        .block_row(systolic_sparse_block_row),
        .block_col(systolic_sparse_block_col),
        .ready(systolic_sparse_ready),
        .act_data(sparse_act_data),
        .act_valid(sparse_act_valid),
        .result_valid(sparse_result_valid),
        .result_data(sparse_result_data),
        .result_block_row(sparse_result_block_row),
        .result_block_col(sparse_result_block_col),
        .result_ready(sparse_result_ready),
        .done(systolic_sparse_done),
        .busy()  // Monitor in perf module
    );
    
    // Block Reorder Buffer (sorts blocks by column for correct GEMM order)
    block_reorder_buffer #(
        .MAX_BLOCKS_PER_ROW(128)
    ) reorder_buffer_inst (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(sparse_result_valid),
        .in_block_col(sparse_result_block_col),
        .in_block_idx({16'h0, sparse_result_block_row}),  // Use row as idx for now
        .in_row_done(systolic_sparse_done),
        .out_valid(reorder_out_valid),
        .out_block_col(reorder_out_block_col),
        .out_block_idx(reorder_out_block_idx),
        .out_row_done(reorder_out_row_done)
    );
    
    // TODO: Add multi_layer_buffer instantiation for multi-layer neural networks
    // TODO: Add BSR metadata BRAMs (row_ptr, col_idx, blocks) - currently stubbed
    
    // BSR Metadata Memory (simple dual-port RAM simulation)
    // In real design, use FPGA BRAMs with separate read/write ports
    reg [31:0] row_ptr_mem [0:255];      // row_ptr: 256 entries × 32 bits
    reg [15:0] col_idx_mem [0:65535];   // col_idx: 64K entries × 16 bits
    // block_mem is now word-addressed (32-bit words)
    reg [31:0] block_mem [0:1048575];   // 1,048,576 words = 4,194,304 bytes
    
    // row_ptr BRAM (DMA writes, scheduler reads)
    always @(posedge clk) begin
        if (dma_row_ptr_we) begin
            row_ptr_mem[dma_row_ptr_waddr] <= dma_row_ptr_wdata;
        end
    end
    assign row_ptr_rd_data = row_ptr_mem[row_ptr_rd_addr];
    
    // col_idx BRAM (DMA writes, scheduler reads)
    always @(posedge clk) begin
        if (dma_col_idx_we) begin
            col_idx_mem[dma_col_idx_waddr] <= dma_col_idx_wdata;
        end
    end
    assign col_idx_rd_data = col_idx_mem[col_idx_rd_addr];
    
    // block BRAM (32-bit word-addressed)
    // DMA writes 32-bit words; Scheduler reads via word address
    always @(posedge clk) begin
        if (dma_block_we) begin
            // dma_block_waddr is a word address (byte_addr >> 2)
            block_mem[dma_block_waddr] <= dma_block_wdata;
        end
    end
    // Scheduler reads 32-bit words directly (word-addressed)
    assign block_rd_data = block_mem[block_rd_addr[31:2]];
    
    // Result ready signal (for now, always ready - add backpressure later)
    assign sparse_result_ready = 1'b1;
    
    // ========================================================================
    // AXI4-Lite Host Interface (Phase 4)
    // ========================================================================
    axi_lite_slave axi_slave_inst (
        .clk(clk),
        .rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awburst(s_axi_awburst),
        .s_axi_awlen(s_axi_awlen),
        .s_axi_awsize(s_axi_awsize),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wlast(s_axi_wlast),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .csr_addr(axi_csr_addr),
        .csr_wen(axi_csr_wen),
        .csr_ren(axi_csr_ren),
        .csr_wdata(axi_csr_wdata),
        .csr_rdata(axi_csr_rdata)
    );
    
    // AXI DMA Bridge - for AXI-Full burst writes to DMA FIFO
    axi_dma_bridge axi_dma_bridge_inst (
        .clk(clk),
        .rst_n(rst_n),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awburst(s_axi_awburst),
        .s_axi_awlen(s_axi_awlen),
        .s_axi_awsize(s_axi_awsize),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wlast(s_axi_wlast),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .dma_fifo_wdata(axi_dma_fifo_wdata),
        .dma_fifo_wen(axi_dma_fifo_wen),
        .dma_fifo_full(axi_dma_fifo_full),
        .dma_fifo_count(axi_dma_fifo_count),
        .axi_error(axi_error),
        .words_written()
    );
    
    // DMA FIFO feedback (TODO: connect to actual FIFO consumer in future)
    // For now, provide safe placeholder values
    assign axi_dma_fifo_full = 1'b0;   // Never full (data accepted but dropped)
    assign axi_dma_fifo_count = 7'd0;  // Always empty
    
    // ========================================================================
    // CSR Mux: UART vs AXI Sources
    // ========================================================================
    // For now, UART has priority. In future, add arbitration.
    assign csr_ren = csr_ren_uart | csr_ren_axi;
    assign csr_wen = csr_wen_uart | csr_wen_axi;
    assign csr_addr = csr_ren_uart ? (rx_addr[7:0]) : (axi_csr_addr);
    assign csr_wdata = csr_wen_uart ? rx_data : axi_csr_wdata;
    assign axi_csr_rdata = csr_rdata;
    
    // UART CSR access
    assign csr_ren_uart = cmd_is_csr_rd;
    assign csr_wen_uart = cmd_is_csr_wr;
    
    // AXI CSR access (mapped from AXI slave)
    assign csr_ren_axi = axi_csr_ren;
    assign csr_wen_axi = axi_csr_wen;
    // Systolic Array (Dense mode - can be disabled if only using sparse)
    systolic_array #(
        .N_ROWS(N_ROWS),
        .N_COLS(N_COLS),
        .PIPE(1),
        .SAT(0)
    ) systolic_inst (
        .clk(clk),
        .rst_n(rst_n),
        .en(array_en),
        .clr(array_clr),
        .load_weight(load_weight),  // NEW: row-stationary weight loading
        .a_in_flat(a_in_flat),
        .b_in_flat(b_in_flat),
        .c_out_flat(c_out_flat)
    );

    // Performance Monitor
    perf #(
        .COUNTER_WIDTH(32)
    ) perf_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start_pulse(start_pulse),
        .done_pulse(done_pulse),
        .busy_signal(busy),
        .total_cycles_count(perf_total_cycles),
        .active_cycles_count(perf_active_cycles),
        .idle_cycles_count(perf_idle_cycles),
        .measurement_done(perf_measurement_done)
    );

    // Activation Buffer
    act_buffer #(
        .TM(TM),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) act_buf_inst (
        .clk(clk),
        .rst_n(rst_n),
        .we(act_we),
        .waddr(act_waddr),
        .wdata(act_wdata),
        .bank_sel_wr(bank_sel_wr_A),
        .rd_en(act_rd_en),
        .k_idx(act_k_idx),
        .bank_sel_rd(bank_sel_rd_A),
        .a_vec(a_vec)
    );

    // Weight Buffer
    wgt_buffer #(
        .TN(TN),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) wgt_buf_inst (
        .clk(clk),
        .rst_n(rst_n),
        .we(wgt_we),
        .waddr(wgt_waddr),
        .wdata(wgt_wdata),
        .bank_sel_wr(bank_sel_wr_B),
        .rd_en(wgt_rd_en),
        .k_idx(wgt_k_idx),
        .bank_sel_rd(bank_sel_rd_B),
        .b_vec(b_vec)
    );

    // Scheduler
    scheduler #(
        .M_W(10),
        .N_W(10),
        .K_W(12),
        .TM_W(3),
        .TN_W(3),
        .TK_W(3),
        .PREPRIME(0),
        .USE_CSR_COUNTS(1),
        .MAX_TM(TM),
        .MAX_TN(TN)
    ) scheduler_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_pulse | cmd_is_start),
        .abort(abort_pulse | cmd_is_abort),
        .M(M[9:0]),
        .N(N[9:0]),
        .K(K[11:0]),
        .Tm(Tm[2:0]),
        .Tn(Tn[2:0]),
        .Tk(Tk[2:0]),
        .MT_csr((M[9:0] != 0) ? ((M[9:0] + Tm[5:0] - 10'd1) / Tm[5:0]) : 10'd1),
        .NT_csr((N[9:0] != 0) ? ((N[9:0] + Tn[5:0] - 10'd1) / Tn[5:0]) : 10'd1),
        .KT_csr((K[11:0] != 0) ? ((K[11:0] + Tk[5:0] - 12'd1) / Tk[5:0]) : 12'd1),
        .valid_A_ping(1'b1),
        .valid_A_pong(1'b1),
        .valid_B_ping(1'b1),
        .valid_B_pong(1'b1),
        .busy(sched_busy),
        .done_tile(sched_done_tile),
        .m_tile(sched_m_tile),
        .n_tile(sched_n_tile),
        .k_tile(sched_k_tile),
        .cycles_tile(cycles_tile),
        .stall_cycles(stall_cycles),
        .rd_en(act_rd_en),
        .k_idx(k_idx_sched),
        .bank_sel_rd_A(bank_sel_rd_A),
        .bank_sel_rd_B(bank_sel_rd_B),
        .clr(array_clr),
        .en(array_en),
        .load_weight(load_weight),  // NEW: row-stationary weight loading control
        .en_mask_row(en_mask_row),
        .en_mask_col(en_mask_col)
    );

    // UART RX
    uart_rx #(
        .DATA_BITS(8),
        .CLK_HZ(CLK_HZ),
        .BAUD(BAUD),
        .OVERSAMPLE(16),
        .PARITY(0),
        .STOP_BITS(1),
        .USE_NCO(0),
        .ACCW(32),
        .MAJORITY3(0)
    ) uart_rx_inst (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_rx(uart_rx),
        .o_data(uart_rx_data),
        .o_valid(uart_rx_valid),
        .o_frm_err(uart_rx_frm_err),
        .o_par_err(uart_rx_par_err)
    );

    // UART TX
    uart_tx #(
        .DATA_BITS(8),
        .CLK_HZ(CLK_HZ),
        .BAUD(BAUD),
        .OVERSAMPLE(16),
        .PARITY(0),
        .STOP_BITS(1),
        .USE_NCO(0),
        .ACCW(32)
    ) uart_tx_inst (
        .i_clk(clk),
        .i_rst_n(rst_n),
        .i_data(uart_tx_data),
        .i_valid(uart_tx_valid),
        .i_ready(uart_tx_ready),
        .o_tx(uart_tx)
    );
    
    // ========================================================================
    // Internal Connections
    // ========================================================================
    
    // Connect systolic array inputs
    assign a_in_flat = a_vec[N_ROWS*8-1:0];
    assign b_in_flat = b_vec[N_COLS*8-1:0];
    
    // Buffer read controls
    assign wgt_rd_en = act_rd_en;
    assign act_k_idx = k_idx_sched[ADDR_WIDTH-1:0];
    assign wgt_k_idx = k_idx_sched[ADDR_WIDTH-1:0];

endmodule

`default_nettype wire
