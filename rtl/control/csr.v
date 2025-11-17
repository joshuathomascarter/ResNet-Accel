// -----------------------------------------------------------------------------
// csr.v — Accel v1 Control/Status (single source of truth)
//  - 32-bit words, little-endian addresses, natural alignment
//  - W1P: start/abort pulses; R/W1C: sticky status; RO: busy & rd-bank
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
`default_nettype none

module csr #(
  parameter ADDR_W = 8  // 256B map
)(
  input  wire              clk,
  input  wire              rst_n,

  // Host CSR bus (from UART bridge or AXI-lite shim)
  input  wire              csr_wen,
  input  wire              csr_ren,
  input  wire [ADDR_W-1:0] csr_addr,   // byte address
  input  wire [31:0]       csr_wdata,
  output reg  [31:0]       csr_rdata,

  // Events from front-end / core
  input  wire              core_busy,
  input  wire              core_done_tile_pulse,
  input  wire              core_bank_sel_rd_A,
  input  wire              core_bank_sel_rd_B,
  input  wire              rx_crc_error,
  input  wire              rx_illegal_cmd,
  // Performance monitor inputs
  input  wire [31:0]       perf_total_cycles,
  input  wire [31:0]       perf_active_cycles,
  input  wire [31:0]       perf_idle_cycles,
  // Systolic array results (captured when done)
  input  wire [127:0]      result_data,  // 4×32-bit results from 2×2 array
  // Pulses / config to core (snapshots consumed by core FSM)
  output wire              start_pulse,
  output wire              abort_pulse,
  output wire              irq_en,

  output wire [31:0]       M, N, K,
  output wire [31:0]       Tm, Tn, Tk,
  output wire [31:0]       m_idx, n_idx, k_idx,
  output wire              bank_sel_wr_A, bank_sel_wr_B, // host-controlled
  output wire              bank_sel_rd_A, bank_sel_rd_B, // RO mirror of core
  output wire [31:0]       Sa_bits, Sw_bits,             // float32 raw
  output wire [31:0]       uart_len_max,
  output wire              uart_crc_en
);

  // Address map (byte offsets)
  localparam CTRL         = 8'h00; // [2]=irq_en (RW), [1]=abort (W1P), [0]=start (W1P)
  localparam DIMS_M       = 8'h04;
  localparam DIMS_N       = 8'h08;
  localparam DIMS_K       = 8'h0C;
  localparam TILES_Tm     = 8'h10;
  localparam TILES_Tn     = 8'h14;
  localparam TILES_Tk     = 8'h18;
  localparam INDEX_m      = 8'h1C;
  localparam INDEX_n      = 8'h20;
  localparam INDEX_k      = 8'h24;
  localparam BUFF         = 8'h28; // [0]=wrA (RW), [1]=wrB (RW), [8]=rdA (RO), [9]=rdB (RO)
  localparam SCALE_Sa     = 8'h2C; // float32 bits
  localparam SCALE_Sw     = 8'h30; // float32 bits
  localparam UART_len_max = 8'h34; // uint32
  localparam UART_crc_en  = 8'h38; // [0]=crc_en
  localparam STATUS       = 8'h3C; // [0]=busy(RO), [1]=done_tile(R/W1C), [8]=err_crc(R/W1C), [9]=err_illegal(R/W1C)
  // Performance monitor registers (Read-Only)
  localparam PERF_TOTAL   = 8'h40; // Total cycles from start to done
  localparam PERF_ACTIVE  = 8'h44; // Cycles where busy was high  
  localparam PERF_IDLE    = 8'h48; // Cycles where busy was low
  // Result registers (Read-Only, captured on done)
  localparam RESULT_0     = 8'h80; // c_out[0]
  localparam RESULT_1     = 8'h84; // c_out[1]
  localparam RESULT_2     = 8'h88; // c_out[2]
  localparam RESULT_3     = 8'h8C; // c_out[3]

  // Backing regs
  reg        r_irq_en;
  reg [31:0] r_M, r_N, r_K;
  reg [31:0] r_Tm, r_Tn, r_Tk;
  reg [31:0] r_m_idx, r_n_idx, r_k_idx;
  reg        r_bank_sel_wr_A, r_bank_sel_wr_B;
  reg [31:0] r_Sa_bits, r_Sw_bits;
  reg [31:0] r_uart_len_max;
  reg        r_uart_crc_en;

  // Sticky status
  reg        st_done_tile;
  reg        st_err_crc;
  reg        st_err_illegal;
  
  // Result capture registers
  reg [31:0] r_result_0, r_result_1, r_result_2, r_result_3;

  // Coverage hooks (for UVM or functional coverage)
  // covergroup cg_csr_write @(posedge clk);
  //   coverpoint csr_addr;
  //   coverpoint csr_wdata;
  // endgroup
  // cg_csr_write cg = new();

  // Reset defaults
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      r_irq_en         <= 1'b0;
      r_M <= 0; r_N <= 0; r_K <= 0;
      r_Tm <= 0; r_Tn <= 0; r_Tk <= 0;
      r_m_idx <= 0; r_n_idx <= 0; r_k_idx <= 0;
      r_bank_sel_wr_A  <= 1'b0;
      r_bank_sel_wr_B  <= 1'b0;
      r_Sa_bits        <= 32'h3F80_0000; // 1.0f
      r_Sw_bits        <= 32'h3F80_0000; // 1.0f
      r_uart_len_max   <= 32'd1024;
      r_uart_crc_en    <= 1'b1;
      st_done_tile     <= 1'b0;
      st_err_crc       <= 1'b0;
      st_err_illegal   <= 1'b0;
      r_result_0       <= 32'd0;
      r_result_1       <= 32'd0;
      r_result_2       <= 32'd0;
      r_result_3       <= 32'd0;
    end else begin
      // Sticky setters
      if (core_done_tile_pulse) st_done_tile <= 1'b1;
      if (rx_crc_error)         st_err_crc   <= 1'b1;
      if (rx_illegal_cmd)       st_err_illegal <= 1'b1;
      
      // Capture results when computation completes
      if (core_done_tile_pulse) begin
        r_result_0 <= result_data[31:0];
        r_result_1 <= result_data[63:32];
        r_result_2 <= result_data[95:64];
        r_result_3 <= result_data[127:96];
      end

      // CSR writes
      if (csr_wen) begin
        unique case (csr_addr)
          CTRL: begin
            r_irq_en <= csr_wdata[2];
            // W1P pulses handled below; readback appears as 0
          end
          DIMS_M:       r_M <= csr_wdata;
          DIMS_N:       r_N <= csr_wdata;
          DIMS_K:       r_K <= csr_wdata;
          TILES_Tm:     r_Tm <= csr_wdata;
          TILES_Tn:     r_Tn <= csr_wdata;
          TILES_Tk:     r_Tk <= csr_wdata;
          INDEX_m:      r_m_idx <= csr_wdata;
          INDEX_n:      r_n_idx <= csr_wdata;
          INDEX_k:      r_k_idx <= csr_wdata;
          BUFF: begin
            r_bank_sel_wr_A <= csr_wdata[0];
            r_bank_sel_wr_B <= csr_wdata[1];
          end
          SCALE_Sa:     r_Sa_bits <= csr_wdata;
          SCALE_Sw:     r_Sw_bits <= csr_wdata;
          UART_len_max: r_uart_len_max <= csr_wdata;
          UART_crc_en:  r_uart_crc_en  <= csr_wdata[0];
          STATUS: begin
            // R/W1C clears
            if (csr_wdata[1]) st_done_tile   <= 1'b0;
            if (csr_wdata[8]) st_err_crc     <= 1'b0;
            if (csr_wdata[9]) st_err_illegal <= 1'b0;
          end
          default: ;
        endcase
      end
    end
  end

  // W1P pulse generation (single-cycle)
  wire w_start = (csr_wen && csr_addr==CTRL && csr_wdata[0]);
  wire w_abort = (csr_wen && csr_addr==CTRL && csr_wdata[1]);

  // Illegal start guard (e.g., zero tiles or start while busy)
  wire dims_illegal  = (r_Tm==0 || r_Tn==0 || r_Tk==0);
  assign start_pulse = w_start && !core_busy && !dims_illegal;
  assign abort_pulse = w_abort;
  // set illegal if blocked
  always @(posedge clk or negedge rst_n) if (rst_n) begin
    if (w_start && (core_busy || dims_illegal)) st_err_illegal <= 1'b1;
  end

  // Read bank selectors are RO mirrors
  assign bank_sel_rd_A = core_bank_sel_rd_A;
  assign bank_sel_rd_B = core_bank_sel_rd_B;

  // Expose config
  assign irq_en         = r_irq_en;
  assign M = r_M;  assign N = r_N;  assign K = r_K;
  assign Tm = r_Tm; assign Tn = r_Tn; assign Tk = r_Tk;
  assign m_idx = r_m_idx; assign n_idx = r_n_idx; assign k_idx = r_k_idx;
  assign bank_sel_wr_A = r_bank_sel_wr_A;
  assign bank_sel_wr_B = r_bank_sel_wr_B;
  assign Sa_bits = r_Sa_bits;  assign Sw_bits = r_Sw_bits;
  assign uart_len_max = r_uart_len_max;
  assign uart_crc_en  = r_uart_crc_en;

  // Read mux (note CTRL start/abort read as 0)
  always @(*) begin
    unique case (csr_addr)
      CTRL:         csr_rdata = {29'b0, r_irq_en, 2'b00};
      DIMS_M:       csr_rdata = r_M;
      DIMS_N:       csr_rdata = r_N;
      DIMS_K:       csr_rdata = r_K;
      TILES_Tm:     csr_rdata = r_Tm;
      TILES_Tn:     csr_rdata = r_Tn;
      TILES_Tk:     csr_rdata = r_Tk;
      INDEX_m:      csr_rdata = r_m_idx;
      INDEX_n:      csr_rdata = r_n_idx;
      INDEX_k:      csr_rdata = r_k_idx;
      BUFF:         csr_rdata = {22'b0,  // keep tidy for future use
                                 6'b0,
                                 bank_sel_rd_B, bank_sel_rd_A,
                                 r_bank_sel_wr_B, r_bank_sel_wr_A};
      SCALE_Sa:     csr_rdata = r_Sa_bits;
      SCALE_Sw:     csr_rdata = r_Sw_bits;
      UART_len_max: csr_rdata = r_uart_len_max;
      UART_crc_en:  csr_rdata = {31'b0, r_uart_crc_en};
      STATUS:       csr_rdata = {22'b0,  // reserved
                                 6'b0,
                                 st_err_illegal, st_err_crc,
                                 st_done_tile, core_busy};
      // Performance monitor registers (Read-Only)
      PERF_TOTAL:   csr_rdata = perf_total_cycles;
      PERF_ACTIVE:  csr_rdata = perf_active_cycles;
      PERF_IDLE:    csr_rdata = perf_idle_cycles;
      // Result registers (Read-Only)
      RESULT_0:     csr_rdata = r_result_0;
      RESULT_1:     csr_rdata = r_result_1;
      RESULT_2:     csr_rdata = r_result_2;
      RESULT_3:     csr_rdata = r_result_3;
      default:      csr_rdata = 32'hDEAD_BEEF;
    endcase
  end

endmodule
`default_nettype wire
