// =============================================================================
// output_accumulator.sv — Double-Buffered Output Accumulator with ReLU/Quantization
// =============================================================================
//
// OVERVIEW
// ========
// This module accumulates partial sums from the systolic array across K-dimension
// tiles, applies optional ReLU activation, quantizes results to INT8, and provides
// a DMA read interface for output transfer to DDR.
//
// WHY ACCUMULATION IS NEEDED
// ==========================
// For large matrix multiplications (e.g., 1024×1024), we tile the K dimension:
//
//   C[M×N] = A[M×K] × B[K×N]
//
// With TK=14 (our tile size), a 1024×1024 matrix requires:
//   1024 / 14 ≈ 74 K-tiles per output tile
//
// Each K-tile produces partial sums that must be accumulated:
//   C_partial_0 = A[0:14] × B[0:14]
//   C_partial_1 = A[14:28] × B[14:28]
//   ...
//   C_final = Σ C_partial_i
//
// DOUBLE-BUFFERING CONCEPT
// ========================
//
//   ┌─────────────────────────────────────────────────────────────────────┐
//   │                    PING-PONG ACCUMULATION                           │
//   ├─────────────────────────────────────────────────────────────────────┤
//   │                                                                      │
//   │   Tile N:    Systolic → [Bank 0] (accumulating)                     │
//   │              DMA ← [Bank 1] (draining previous result)              │
//   │                                                                      │
//   │   Tile N+1:  Systolic → [Bank 1] (accumulating)                     │
//   │              DMA ← [Bank 0] (draining previous result)              │
//   │                                                                      │
//   └─────────────────────────────────────────────────────────────────────┘
//
// MEMORY ORGANIZATION
// ===================
// Each bank stores a complete output tile:
//   - 196 accumulators (14×14 outputs)
//   - Each accumulator is 32 bits (INT32 for overflow headroom)
//   - Total per bank: 196 × 32 = 6,272 bits ≈ 0.8 KB
//
// WHY INT32 ACCUMULATORS?
//   - INT8 × INT8 = INT16 product
//   - Summing 1024 INT16 products requires INT32 to prevent overflow
//   - Max value: 127 × 127 × 1024 = 16,516,096 (fits in 32 bits)
//
// OUTPUT PIPELINE
// ===============
// The DMA read path includes ReLU and quantization:
//
//   ┌────────────┐    ┌────────┐    ┌────────────┐    ┌──────────┐
//   │ INT32 Acc  │ → │  ReLU  │ → │ Scale      │ → │  Saturate │ → INT8 Out
//   │ (32 bits)  │    │ max(0,x)│    │ ×scale/2^16│    │ [-128,127]│
//   └────────────┘    └────────┘    └────────────┘    └──────────┘
//
// The scale_factor uses Q16.16 fixed-point format:
//   output = (acc × scale) >> 16
//
// MAGIC NUMBERS
// =============
// 196 = 14×14 accumulators (matches systolic array output size)
// 32  = Accumulator bit width (prevents overflow for K up to 65536)
// 8   = Output bit width (INT8 for next layer input)
// 16  = Fixed-point shift for Q16.16 scale factor
// 127, -128 = INT8 saturation bounds
//
// TIMING
// ======
// - Accumulation: 1 cycle (combinational add + register)
// - DMA read: 2 cycles (read from bank + ReLU/quantize)
//
// =============================================================================

`default_nettype none

module output_accumulator #(
    // =========================================================================
    // PARAMETER: N_ROWS - Systolic Array Row Count
    // =========================================================================
    // Must match systolic_array N_ROWS parameter.
    // Determines number of output rows per tile.
    parameter N_ROWS    = 14,
    
    // =========================================================================
    // PARAMETER: N_COLS - Systolic Array Column Count
    // =========================================================================
    // Must match systolic_array N_COLS parameter.
    // Total accumulators = N_ROWS × N_COLS = 196.
    parameter N_COLS    = 14,
    
    // =========================================================================
    // PARAMETER: ACC_W - Accumulator Bit Width
    // =========================================================================
    // 32 bits provides overflow protection for K-dimension up to 65536.
    // For INT8 inputs: max_acc = 127² × K ≤ 2^31 when K ≤ 133,169.
    parameter ACC_W     = 32,
    
    // =========================================================================
    // PARAMETER: OUT_W - Output Data Width
    // =========================================================================
    // 8 bits for INT8 quantized output to next layer.
    // Saturated to [-128, 127] during quantization.
    parameter OUT_W     = 8,
    
    // =========================================================================
    // PARAMETER: ADDR_W - DMA Address Width
    // =========================================================================
    // 10 bits addresses up to 1024 64-bit words.
    // For 196 accumulators: need 196/8 = 25 words.
    parameter ADDR_W    = 10
)(
    // =========================================================================
    // SYSTEM INTERFACE
    // =========================================================================
    input  wire                         clk,
    input  wire                         rst_n,

    // =========================================================================
    // CONTROL INTERFACE (from Scheduler/CSR)
    // =========================================================================
    
    /** acc_valid - Systolic array output valid this cycle */
    input  wire                         acc_valid,
    
    /** acc_clear - Clear accumulators for new tile (pulse) */
    input  wire                         acc_clear,
    
    /** tile_done - Current tile complete, swap banks (pulse) */
    input  wire                         tile_done,
    
    /** relu_en - Apply ReLU activation (max(0,x)) before output */
    input  wire                         relu_en,
    
    /** 
     * scale_factor - Quantization scale in Q16.16 fixed-point
     * --------------------------------------------------------
     * Used to requantize INT32 accumulators to INT8.
     * Calculated during quantization-aware training.
     * Formula: output = (acc × scale) >> 16
     * Example: scale = 0x00010000 = 1.0 (no scaling)
     */
    input  wire [31:0]                  scale_factor,
    
    // =========================================================================
    // SYSTOLIC ARRAY INPUT
    // =========================================================================
    /**
     * systolic_out - Flattened partial sum output from systolic array
     * ----------------------------------------------------------------
     * Layout: [row0_col0, row0_col1, ..., row13_col13]
     * Each element is ACC_W bits (32 bits).
     * Total width: N_ROWS × N_COLS × ACC_W = 14 × 14 × 32 = 6,272 bits
     */
    input  wire [N_ROWS*N_COLS*ACC_W-1:0] systolic_out,

    // =========================================================================
    // DMA READ INTERFACE (to output DMA)
    // =========================================================================
    
    /** dma_rd_en - DMA read enable */
    input  wire                         dma_rd_en,
    
    /** 
     * dma_rd_addr - DMA read address (64-bit word index)
     * ---------------------------------------------------
     * Each address reads 8 consecutive INT8 outputs.
     * Address 0 → outputs[0:7], Address 1 → outputs[8:15], etc.
     */
    input  wire [ADDR_W-1:0]            dma_rd_addr,
    
    /** dma_rd_data - Read data (8 INT8 values packed in 64 bits) */
    output reg  [63:0]                  dma_rd_data,
    
    /** dma_ready - Inactive bank ready for DMA read */
    output wire                         dma_ready,

    // =========================================================================
    // STATUS
    // =========================================================================
    
    /** busy - Accumulation in progress */
    output reg                          busy,
    
    /** bank_sel - Current active bank (0 or 1) */
    output reg                          bank_sel,
    
    /** acc_debug - Debug: value of first accumulator in active bank */
    output wire [31:0]                  acc_debug
);

    // =========================================================================
    // LOCAL PARAMETERS
    // =========================================================================
    localparam NUM_ACCS = N_ROWS * N_COLS;  // 196 for 14x14 array
    localparam BANK_DEPTH = NUM_ACCS;       // One accumulator per output
    
    // =========================================================================
    // DOUBLE-BUFFERED ACCUMULATOR BANKS
    // =========================================================================
    /**
     * Two banks of 196 × 32-bit signed accumulators.
     * One bank accumulates while the other is read by DMA.
     * 
     * Memory usage: 2 × 196 × 32 = 12,544 bits ≈ 1.5 KB
     * Fits in distributed RAM or small BRAM.
     */
    reg signed [ACC_W-1:0] acc_bank0 [0:BANK_DEPTH-1];
    reg signed [ACC_W-1:0] acc_bank1 [0:BANK_DEPTH-1];
    
    // Bank ready flags - set when tile completes, cleared when DMA reads
    reg bank0_ready;
    reg bank1_ready;
    
    // DMA reads from inactive bank (opposite of bank_sel)
    assign dma_ready = bank_sel ? bank0_ready : bank1_ready;

    // =========================================================================
    // UNPACK SYSTOLIC ARRAY OUTPUT
    // =========================================================================
    /**
     * Convert flattened bus to array of 32-bit values.
     * This is purely combinational (no registers).
     */
    wire signed [ACC_W-1:0] sys_out [0:NUM_ACCS-1];
    
    genvar i;
    generate
        for (i = 0; i < NUM_ACCS; i = i + 1) begin : UNPACK
            assign sys_out[i] = systolic_out[i*ACC_W +: ACC_W];
        end
    endgenerate

    // =========================================================================
    // ACCUMULATION LOGIC
    // =========================================================================
    /**
     * Main accumulation state machine.
     * 
     * Operations:
     * 1. acc_clear: Zero all accumulators in active bank (new tile start)
     * 2. acc_valid: Add systolic outputs to active bank accumulators
     * 3. tile_done: Swap banks, mark completed bank ready for DMA
     * 4. dma_rd_en: Clear ready flag (DMA has started reading)
     */
    integer j;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // ─────────────────────────────────────────────────────────────────
            // RESET STATE
            // ─────────────────────────────────────────────────────────────────
            bank_sel <= 1'b0;
            bank0_ready <= 1'b0;
            bank1_ready <= 1'b0;
            busy <= 1'b0;
            
            // Clear both banks to zero
            for (j = 0; j < BANK_DEPTH; j = j + 1) begin
                acc_bank0[j] <= 32'sd0;
                acc_bank1[j] <= 32'sd0;
            end
        end else begin
            // ─────────────────────────────────────────────────────────────────
            // CLEAR: Zero accumulators for new output tile
            // ─────────────────────────────────────────────────────────────────
            // Must be done before first K-tile arrives.
            // Only clears active bank (other bank may be draining).
            if (acc_clear) begin
                if (bank_sel == 1'b0) begin
                    for (j = 0; j < BANK_DEPTH; j = j + 1) begin
                        acc_bank0[j] <= 32'sd0;
                    end
                end else begin
                    for (j = 0; j < BANK_DEPTH; j = j + 1) begin
                        acc_bank1[j] <= 32'sd0;
                    end
                end
                busy <= 1'b1;
            end
            
            // ─────────────────────────────────────────────────────────────────
            // ACCUMULATE: Add partial sums from systolic array
            // ─────────────────────────────────────────────────────────────────
            // All 196 accumulators update in parallel (massive parallelism).
            // This is the performance-critical path.
            if (acc_valid) begin
                if (bank_sel == 1'b0) begin
                    for (j = 0; j < BANK_DEPTH; j = j + 1) begin
                        acc_bank0[j] <= acc_bank0[j] + sys_out[j];
                    end
                end else begin
                    for (j = 0; j < BANK_DEPTH; j = j + 1) begin
                        acc_bank1[j] <= acc_bank1[j] + sys_out[j];
                    end
                end
            end
            
            // ─────────────────────────────────────────────────────────────────
            // TILE DONE: Swap banks for ping-pong operation
            // ─────────────────────────────────────────────────────────────────
            // After all K-tiles for an output tile are processed.
            // Completed bank becomes available for DMA.
            if (tile_done) begin
                bank_sel <= ~bank_sel;
                busy <= 1'b0;
                
                // Mark completed bank as ready for DMA readout
                if (bank_sel == 1'b0) begin
                    bank0_ready <= 1'b1;
                end else begin
                    bank1_ready <= 1'b1;
                end
            end
            
            // ─────────────────────────────────────────────────────────────────
            // DMA START: Clear ready flag to prevent double-read
            // ─────────────────────────────────────────────────────────────────
            if (dma_rd_en) begin
                if (bank_sel == 1'b1) begin  // DMA reads from bank0 when bank_sel=1
                    bank0_ready <= 1'b0;
                end else begin               // DMA reads from bank1 when bank_sel=0
                    bank1_ready <= 1'b0;
                end
            end
        end
    end

    // =========================================================================
    // DMA READ PATH - Pipeline Stage 1: Bank Read
    // =========================================================================
    /**
     * Read 8 consecutive accumulators from inactive bank.
     * 
     * Address mapping:
     *   dma_rd_addr[6:0] = 64-bit word address
     *   {dma_rd_addr, 3'bXXX} = accumulator index (8 per word)
     * 
     * Example: dma_rd_addr=5 reads accumulators 40-47
     */
    reg signed [ACC_W-1:0] rd_acc_0, rd_acc_1, rd_acc_2, rd_acc_3;
    reg signed [ACC_W-1:0] rd_acc_4, rd_acc_5, rd_acc_6, rd_acc_7;
    reg [ADDR_W-1:0] rd_addr_d1;
    reg rd_valid_d1;
    
    always @(posedge clk) begin
        rd_valid_d1 <= dma_rd_en;
        rd_addr_d1 <= dma_rd_addr;
        
        if (dma_rd_en) begin
            // Read 8 consecutive accumulators for 64-bit output
            // Note: bank_sel=1 means we accumulate to bank1, read from bank0
            if (bank_sel == 1'b1) begin  // Read from bank 0
                rd_acc_0 <= acc_bank0[{dma_rd_addr, 3'b000}];
                rd_acc_1 <= acc_bank0[{dma_rd_addr, 3'b001}];
                rd_acc_2 <= acc_bank0[{dma_rd_addr, 3'b010}];
                rd_acc_3 <= acc_bank0[{dma_rd_addr, 3'b011}];
                rd_acc_4 <= acc_bank0[{dma_rd_addr, 3'b100}];
                rd_acc_5 <= acc_bank0[{dma_rd_addr, 3'b101}];
                rd_acc_6 <= acc_bank0[{dma_rd_addr, 3'b110}];
                rd_acc_7 <= acc_bank0[{dma_rd_addr, 3'b111}];
            end else begin               // Read from bank 1
                rd_acc_0 <= acc_bank1[{dma_rd_addr, 3'b000}];
                rd_acc_1 <= acc_bank1[{dma_rd_addr, 3'b001}];
                rd_acc_2 <= acc_bank1[{dma_rd_addr, 3'b010}];
                rd_acc_3 <= acc_bank1[{dma_rd_addr, 3'b011}];
                rd_acc_4 <= acc_bank1[{dma_rd_addr, 3'b100}];
                rd_acc_5 <= acc_bank1[{dma_rd_addr, 3'b101}];
                rd_acc_6 <= acc_bank1[{dma_rd_addr, 3'b110}];
                rd_acc_7 <= acc_bank1[{dma_rd_addr, 3'b111}];
            end
        end
    end

    // =========================================================================
    // DMA READ PATH - Pipeline Stage 2: ReLU + Quantization
    // =========================================================================
    /**
     * quantize_relu - Apply ReLU activation and requantize to INT8
     * 
     * Pipeline:
     *   1. ReLU: if (relu_en && value < 0) → 0
     *   2. Scale: value × scale >> 16 (Q16.16 fixed-point)
     *   3. Saturate: clamp to [-128, 127] range
     * 
     * @param acc_val  Input accumulator value (signed 32-bit)
     * @param scale    Quantization scale factor (Q16.16)
     * @param relu     Enable ReLU activation
     * @return         Quantized INT8 output
     */
    function automatic [OUT_W-1:0] quantize_relu;
        input signed [ACC_W-1:0] acc_val;
        input [31:0] scale;
        input relu;
        
        reg signed [63:0] scaled;
        reg signed [ACC_W-1:0] relu_val;
        reg signed [15:0] quant_val;
    begin
        // Step 1: ReLU activation (max(0, x))
        // Common in CNNs - introduces non-linearity
        if (relu && acc_val < 0)
            relu_val = 32'sd0;
        else
            relu_val = acc_val;
        
        // Step 2: Fixed-point scaling
        // Q16.16 format: lower 16 bits are fraction
        // Result shifted right by 16 to extract integer part
        scaled = (relu_val * $signed({1'b0, scale[15:0]})) >>> 16;
        
        // Step 3: Saturate to INT8 range
        // Prevents overflow when result exceeds INT8 capacity
        // Magic numbers: 127 = max INT8, -128 = min INT8
        if (scaled > 127)
            quant_val = 127;
        else if (scaled < -128)
            quant_val = -128;
        else
            quant_val = scaled[15:0];
        
        quantize_relu = quant_val[OUT_W-1:0];
    end
    endfunction

    // Apply ReLU and quantization to all 8 accumulators
    always @(posedge clk) begin
        if (rd_valid_d1) begin
            // Pack 8 INT8 values into 64-bit output
            // Byte order: [7][6][5][4][3][2][1][0] (little-endian)
            dma_rd_data <= {
                quantize_relu(rd_acc_7, scale_factor, relu_en),
                quantize_relu(rd_acc_6, scale_factor, relu_en),
                quantize_relu(rd_acc_5, scale_factor, relu_en),
                quantize_relu(rd_acc_4, scale_factor, relu_en),
                quantize_relu(rd_acc_3, scale_factor, relu_en),
                quantize_relu(rd_acc_2, scale_factor, relu_en),
                quantize_relu(rd_acc_1, scale_factor, relu_en),
                quantize_relu(rd_acc_0, scale_factor, relu_en)
            };
        end
    end

    // =========================================================================
    // DEBUG OUTPUT
    // =========================================================================
    // Expose first accumulator for waveform debugging
    assign acc_debug = bank_sel ? acc_bank1[0] : acc_bank0[0];

endmodule

`default_nettype wire
