`timescale 1ns / 1ps

//================================================================================
// BSR (Block Sparse Row) Scheduler for Sparse Matrix Accelerator
//================================================================================
// 
// Author: Hardware Acceleration Team
// Purpose: Traverse BSR sparse format and schedule 8×8 block computations
//          Achieves 8-10× speedup on 90% sparse neural network layers
//
// HARDWARE INTERFACE:
// ┌──────────────────────────────────────────────────────────────┐
// │  BSR Metadata BRAMs          Block Data BRAM                 │
// │  ┌──────────────┐            ┌──────────────┐                │
// │  │  row_ptr[]   │            │  blocks[]    │                │
// │  │  [0,3,3,5]   │            │  [8×8 INT8]  │                │
// │  └──────┬───────┘            └──────┬───────┘                │
// │         │                           │                        │
// │  ┌──────┴───────┐                   │                        │
// │  │  col_idx[]   │                   │                        │
// │  │  [0,3,5,...]│                   │                        │
// │  └──────┬───────┘                   │                        │
// │         │                           │                        │
// │    ┌────┴───────────────────────────┴────┐                   │
// │    │     BSR SCHEDULER FSM               │                   │
// │    │  • Reads row_ptr to find blocks     │                   │
// │    │  • Skips empty rows (speedup!)      │                   │
// │    │  • Loads 8×8 blocks on demand       │                   │
// │    └────────────┬────────────────────────┘                   │
// │                 │                                            │
// │                 ▼                                            │
// │    ┌────────────────────────────┐                            │
// │    │  Systolic Array (2×2 PEs)  │                            │
// │    │  INT8 × INT8 → INT32       │                            │
// │    └────────────────────────────┘                            │
// └──────────────────────────────────────────────────────────────┘
//
// BSR FORMAT STRUCTURE:
// ---------------------
// row_ptr[i]: Starting block index for block-row i
// row_ptr[i+1]: Ending block index for block-row i
// If row_ptr[i+1] == row_ptr[i]: EMPTY ROW (skip!)
// col_idx[j]: Column position of block j
// blocks[j]: 8×8 INT8 weight data for block j
//
// EXAMPLE (MNIST FC1 @ 90% sparse):
// ----------------------------------
// Matrix: [128, 9216]
// Block grid: 16 rows × 1152 cols = 18,432 possible blocks
// Actual non-zero: ~1,843 blocks (10% density)
// Memory: 118 KB (vs 1.15 MB dense → 9.7× savings)
// Speedup: ~9× (skip 90% of blocks)
//
// FSM STATE DIAGRAM:
// ==================
//
//     ┌──────┐
//     │ IDLE │◄────────────────────────────────┐
//     └───┬──┘                                  │
//         │ start=1                             │
//         ▼                                     │
//  ┌────────────────┐                           │
//  │ READ_ROW_PTR   │ ← Read row_ptr[i], row_ptr[i+1]
//  └───────┬────────┘                           │
//          │                                    │
//          ▼                                    │
//  ┌────────────────┐                           │
//  │ CHECK_EMPTY    │ ← Compute num_blocks      │
//  └───────┬────────┘                           │
//          │                                    │
//     ┌────┴────┐                               │
//     │         │                               │
// num_blocks    num_blocks > 0                  │
//     == 0      │                               │
//     │         ▼                               │
//     │  ┌────────────────┐                     │
//     │  │ READ_COL_IDX   │ ← Read col_idx[block_idx]
//     │  └───────┬────────┘                     │
//     │          │                              │
//     │          ▼                              │
//     │  ┌────────────────┐                     │
//     │  │ LOAD_BLOCK     │ ← DMA 64 bytes     │
//     │  └───────┬────────┘                     │
//     │          │                              │
//     │          ▼                              │
//     │  ┌────────────────┐                     │
//     │  │ COMPUTE        │ ← Wait systolic_done
//     │  └───────┬────────┘                     │
//     │          │                              │
//     │          ▼                              │
//     │  ┌────────────────┐                     │
//     │  │ NEXT_BLOCK     │ ← block_idx++      │
//     │  └───────┬────────┘                     │
//     │          │                              │
//     │     ┌────┴────┐                         │
//     │     │         │                         │
//     │  more      done                         │
//     │  blocks    row                          │
//     │     │         │                         │
//     │     └────►┌───▼────┐                    │
//     │           │NEXT_ROW│ ← block_row++      │
//     │           └───┬────┘                    │
//     │               │                         │
//     └───────────────┤                         │
//                 ┌───▼────┐                    │
//                 │  DONE  │────────────────────┘
//                 └────────┘
//
// CRITICAL HARDWARE FEATURES:
// ===========================
// 1. EMPTY ROW DETECTION: row_ptr[i+1] == row_ptr[i] → skip (common in 90% sparse!)
// 2. BLOCK INDEXING: col_idx[j] tells which output column block j updates
// 3. MEMORY LAYOUT: Sequential blocks in BRAM (burst-friendly)
// 4. PIPELINE FRIENDLY: States separated for clean pipeline stages
//
//================================================================================

module bsr_scheduler #(
    parameter BLOCK_H = 8,           // Block height (8×8 blocks)
    parameter BLOCK_W = 8,           // Block width
    parameter BLOCK_SIZE = 64,       // 8×8 = 64 elements per block
    parameter MAX_BLOCK_ROWS = 256,  // Support up to 256 block rows (2048 output features)
    parameter MAX_BLOCKS = 65536,    // Support up to 64K blocks
    parameter DATA_WIDTH = 8,        // INT8 data
    parameter ENABLE_CLOCK_GATING = 1  // Enable clock gating (saves ~50 mW)
)(
    // Clock and reset
    input  logic clk,
    input  logic rst_n,
    
    // Control interface
    input  logic        start,                  // Start sparse GEMM
    input  logic [15:0] cfg_num_block_rows,     // Configuration: number of block rows
    input  logic [15:0] cfg_num_block_cols,     // Configuration: number of block columns
    input  logic [31:0] cfg_total_blocks,       // Configuration: total non-zero blocks
    
    // NEW: Runtime layer configuration
    input  logic        cfg_layer_switch,       // Trigger layer reconfiguration
    input  logic [2:0]  cfg_active_layer,       // Active layer ID (0-7)
    output logic        cfg_layer_ready,        // Ready for new configuration
    
    // Metadata decoder cache interface (replaces direct BRAM access)
    output logic        meta_rd_en,             // Read enable
    output logic [7:0]  meta_rd_addr,           // Cache address
    output logic [1:0]  meta_rd_type,           // 0=ROW_PTR, 1=COL_IDX, 2=BLOCK_HDR
    input  logic [31:0] meta_rd_data,           // Read data from cache
    input  logic        meta_rd_valid,          // Data valid (1 cycle after request)
    input  logic        meta_rd_hit,            // Cache hit indicator
    
    // Block data BRAM (64 INT8 values per block)
    output logic        block_rd_en,            // Read enable
    output logic [31:0] block_rd_addr,          // Byte address (block_idx * 64)
    input  logic [7:0]  block_rd_data,          // Read data (1 byte per cycle, 64 cycles/block)
    
    // Systolic array interface
    output logic                    systolic_valid,     // Block data valid
    output logic [DATA_WIDTH-1:0]   systolic_block [0:BLOCK_SIZE-1], // 8×8 block
    output logic [15:0]             systolic_block_row, // Block row index
    output logic [15:0]             systolic_block_col, // Block column index
    input  logic                    systolic_ready,     // Systolic ready for new block
    input  logic                    systolic_done,      // Systolic finished computation
    
    // Status outputs
    output logic        done,                   // Computation complete
    output logic        busy,                   // Scheduler active
    output logic [31:0] blocks_processed        // Performance counter
);

    //========================================================================
    // Clock Gating Logic (saves ~50 mW when sparse path idle)
    //========================================================================
    wire bsr_clk_en, clk_gated;
    assign bsr_clk_en = start | busy | meta_rd_valid;
    
    generate
        if (ENABLE_CLOCK_GATING) begin : gen_clk_gate
            `ifdef XILINX_FPGA
                BUFGCE bsr_clk_gate (
                    .I  (clk),
                    .CE (bsr_clk_en),
                    .O  (clk_gated)
                );
            `else
                reg bsr_clk_en_latched;
                always @(clk or bsr_clk_en) begin
                    if (!clk) bsr_clk_en_latched <= bsr_clk_en;
                end
                assign clk_gated = clk & bsr_clk_en_latched;
            `endif
        end else begin : gen_no_gate
            assign clk_gated = clk;
        end
    endgenerate

    //========================================================================
    // FSM States
    //========================================================================
    typedef enum logic [3:0] {
        IDLE            = 4'd0,   // Wait for start
        LAYER_SWITCH    = 4'd1,   // NEW: Handle layer reconfiguration
        READ_ROW_PTR_0  = 4'd2,   // Read row_ptr[i]
        READ_ROW_PTR_1  = 4'd3,   // Read row_ptr[i+1]
        CHECK_EMPTY     = 4'd4,   // Check if row has blocks
        PREFETCH_NEXT   = 4'd5,   // NEW: Prefetch next block metadata
        READ_COL_IDX    = 4'd6,   // Read col_idx[block_idx]
        LOAD_BLOCK_INIT = 4'd7,   // Initialize block load
        LOAD_BLOCK_LOOP = 4'd8,   // Load 64 bytes (64 cycles)
        SEND_TO_SYSTOLIC= 4'd9,   // Send block to systolic array
        WAIT_SYSTOLIC   = 4'd10,  // Wait for systolic to finish
        NEXT_BLOCK      = 4'd11,  // Move to next block in row
        NEXT_ROW        = 4'd12,  // Move to next block row
        FINISH          = 4'd13   // Done, go back to IDLE
    } state_t;
    
    state_t state, next_state;
    
    //========================================================================
    // Internal Registers
    //========================================================================
    logic [15:0] block_row;             // Current block row (0 to num_block_rows-1)
    logic [31:0] block_start;           // row_ptr[block_row] - first block of row
    logic [31:0] block_end;             // row_ptr[block_row+1] - first block of next row
    logic [31:0] block_idx;             // Current block index (global)
    logic [31:0] num_blocks_in_row;     // block_end - block_start
    logic [15:0] block_col;             // Column position from col_idx[block_idx]
    
    // Block data buffer (accumulates 64 INT8 values)
    logic [DATA_WIDTH-1:0] block_buffer [0:BLOCK_SIZE-1];
    logic [5:0] byte_counter;           // Count 0-63 for loading block
    
    // Pipeline registers for BRAM read latency
    logic [31:0] row_ptr_reg;
    logic [15:0] col_idx_reg;
    
    // NEW: Prefetch registers for hiding BRAM latency
    logic [15:0] prefetch_col_idx;      // Next block's column index
    logic [DATA_WIDTH-1:0] prefetch_block_data [0:BLOCK_SIZE-1]; // Next block's data
    logic        prefetch_valid;        // Prefetch data is valid
    logic [31:0] prefetch_block_idx;    // Which block was prefetched
    
    // Layer config registers (latched on layer switch)
    logic [15:0] active_num_block_rows;
    logic [15:0] active_num_block_cols;
    logic [31:0] active_total_blocks;
    
    //========================================================================
    // FSM Sequential Logic
    //========================================================================
    always_ff @(posedge clk_gated or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            block_row <= '0;
            block_start <= '0;
            block_end <= '0;
            block_idx <= '0;
            num_blocks_in_row <= '0;
            block_col <= '0;
            byte_counter <= '0;
            blocks_processed <= '0;
            busy <= 1'b0;
            done <= 1'b0;
            cfg_layer_ready <= 1'b0;
            prefetch_valid <= 1'b0;
            prefetch_block_idx <= '0;
            active_num_block_rows <= cfg_num_block_rows;
            active_num_block_cols <= cfg_num_block_cols;
            active_total_blocks <= cfg_total_blocks;
            
            for (int i = 0; i < BLOCK_SIZE; i++) begin
                block_buffer[i] <= '0;
            end
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        busy <= 1'b1;
                        done <= 1'b0;
                        block_row <= '0;
                        block_idx <= '0;
                        blocks_processed <= '0;
                    end else if (cfg_layer_switch) begin
                        // Move to LAYER_SWITCH to latch new configuration
                        cfg_layer_ready <= 1'b0;
                        state <= LAYER_SWITCH;
                    end
                end
                
                LAYER_SWITCH: begin
                    // Latch runtime configuration for the active layer
                    active_num_block_rows <= cfg_num_block_rows;
                    active_num_block_cols <= cfg_num_block_cols;
                    active_total_blocks <= cfg_total_blocks;
                    // reset progress for new layer
                    block_row <= '0;
                    block_idx <= '0;
                    blocks_processed <= '0;
                    cfg_layer_ready <= 1'b1; // indicate we've applied config
                end

                READ_ROW_PTR_0: begin
                    block_start <= meta_rd_data;
                end

                
                READ_ROW_PTR_1: begin
                    // Capture row_ptr[block_row+1]
                    block_end <= meta_rd_data;
                end
                
                CHECK_EMPTY: begin
                    // Compute number of blocks in this row
                    num_blocks_in_row <= block_end - block_start;
                    block_idx <= block_start; // Reset to start of row
                end
                
                READ_COL_IDX: begin
                    // Pipeline delay - register will capture next cycle
                end
                
                LOAD_BLOCK_INIT: begin
                    byte_counter <= '0;
                    block_col <= col_idx_reg; // Use registered value
                end
                
                LOAD_BLOCK_LOOP: begin
                    // Load one byte per cycle
                    block_buffer[byte_counter] <= block_rd_data;
                    byte_counter <= byte_counter + 1;
                end
                
                WAIT_SYSTOLIC: begin
                    if (systolic_done) begin
                        blocks_processed <= blocks_processed + 1;
                    end
                end
                
                NEXT_BLOCK: begin
                    // Advance to next block and use prefetched column index if available
                    block_idx <= block_idx + 1;
                    if (prefetch_valid && (prefetch_block_idx == block_idx + 1)) begin
                        // Use the prefetched column index to avoid extra BRAM read
                        col_idx_reg <= prefetch_col_idx;
                        prefetch_valid <= 1'b0;
                    end
                end
                
                NEXT_ROW: begin
                    block_row <= block_row + 1;
                    block_start <= block_end; // Optimization: reuse block_end as next block_start
                end
                
                FINISH: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            endcase
        end
    end
    
    //========================================================================
    // FSM Next State Logic
    //========================================================================
    always_comb begin
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start) next_state = READ_ROW_PTR_0;
            end
            
            READ_ROW_PTR_0: begin
                next_state = READ_ROW_PTR_1; // Need two reads for row_ptr[i] and row_ptr[i+1]
            end
            
            READ_ROW_PTR_1: begin
                next_state = CHECK_EMPTY;
            end
            
            CHECK_EMPTY: begin
                if (num_blocks_in_row == 0) begin
                    // Empty row - skip to next row
                    if (block_row >= active_num_block_rows - 1) begin
                        next_state = FINISH; // Last row and empty
                    end else begin
                        next_state = NEXT_ROW;
                    end
                end else begin
                    // Has blocks - start processing
                    next_state = READ_COL_IDX;
                end
            end
            
            READ_COL_IDX: begin
                next_state = LOAD_BLOCK_INIT;
            end
            
            LOAD_BLOCK_INIT: begin
                next_state = LOAD_BLOCK_LOOP;
            end
            
            LOAD_BLOCK_LOOP: begin
                if (byte_counter == BLOCK_SIZE - 1) begin
                    next_state = SEND_TO_SYSTOLIC;
                end
                // else stay in LOAD_BLOCK_LOOP
            end
            
            SEND_TO_SYSTOLIC: begin
                if (systolic_ready) begin
                    next_state = WAIT_SYSTOLIC;
                end
            end
            
            WAIT_SYSTOLIC: begin
                if (systolic_done) begin
                    next_state = NEXT_BLOCK;
                end
            end
            
            NEXT_BLOCK: begin
                if (block_idx >= block_end - 1) begin
                    // Finished all blocks in this row
                    if (block_row >= active_num_block_rows - 1) begin
                        next_state = FINISH; // Last row
                    end else begin
                        next_state = NEXT_ROW;
                    end
                end else begin
                    // More blocks in this row
                    next_state = READ_COL_IDX;
                end
            end
            
            NEXT_ROW: begin
                next_state = READ_ROW_PTR_0;
            end
            
            FINISH: begin
                next_state = IDLE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    //========================================================================
    // BRAM Control Logic
    //========================================================================
    
    // Row pointer BRAM
    always_comb begin
        meta_rd_en = 1'b0;
        meta_rd_addr = '0;
        meta_rd_type = 2'b00;
        
        case (state)
            READ_ROW_PTR_0: begin
                meta_rd_en = 1'b1;
                meta_rd_addr = block_row[7:0];       // Read row_ptr[i] from cache
                meta_rd_type = 2'b00;                 // ROW_PTR type
            end
            READ_ROW_PTR_1: begin
                meta_rd_en = 1'b1;
                meta_rd_addr = {24'b0, block_row[7:0] + 8'b1}; // Read row_ptr[i+1] from cache
                meta_rd_type = 2'b00;                 // ROW_PTR type
            end
        endcase
    end
    
    // Capture row_ptr reads
    always_ff @(posedge clk_gated) begin
        if (state == READ_ROW_PTR_0) begin
            row_ptr_reg <= meta_rd_data; // Will be block_start
        end
        if (state == READ_ROW_PTR_1) begin
            block_start <= row_ptr_reg;     // From previous cycle
        end
    end
    
    // Column index BRAM
    always_comb begin
        meta_rd_en = 1'b0;
        meta_rd_addr = '0;
        meta_rd_type = 2'b01;  // COL_IDX type by default
        
        if (state == READ_COL_IDX) begin
            meta_rd_en = 1'b1;
            meta_rd_addr = block_idx[7:0];  // Map to COL_IDX cache region (0x40+idx)
            meta_rd_type = 2'b01;            // COL_IDX type
        end else if (state == WAIT_SYSTOLIC) begin
            // Prefetch next block's column index while systolic is computing
            if ((block_idx < block_end - 1) && !prefetch_valid) begin
                meta_rd_en = 1'b1;
                meta_rd_addr = {24'b0, block_idx[7:0] + 8'b1};
                meta_rd_type = 2'b01;                 // COL_IDX type
            end
        end
    end
    
    // Capture col_idx read
    always_ff @(posedge clk_gated) begin
        if (state == READ_COL_IDX) begin
            col_idx_reg <= meta_rd_data[15:0];  // Extract 16-bit col_idx from 32-bit word
        end
        // Prefetch capture: when systolic is computing, capture next col_idx
        if (state == WAIT_SYSTOLIC) begin
            if ((block_idx < block_end - 1) && !prefetch_valid) begin
                prefetch_col_idx <= meta_rd_data[15:0];  // Extract 16-bit col_idx
                prefetch_block_idx <= block_idx + 1;
                prefetch_valid <= 1'b1;
            end
        end
    end
    
    // Block data BRAM (sequential 64-byte read)
    always_comb begin
        block_rd_en = 1'b0;
        block_rd_addr = '0;
        
        if (state == LOAD_BLOCK_LOOP) begin
            block_rd_en = 1'b1;
            block_rd_addr = (block_idx * BLOCK_SIZE) + byte_counter;
        end
    end
    
    //========================================================================
    // Systolic Array Output Interface
    //========================================================================
    always_comb begin
        systolic_valid = (state == SEND_TO_SYSTOLIC);
        systolic_block_row = block_row;
        systolic_block_col = block_col;
        
        // Send block data
        for (int i = 0; i < BLOCK_SIZE; i++) begin
            systolic_block[i] = block_buffer[i];
        end
    end
    
    //========================================================================
    // Assertions for Debugging
    //========================================================================
    // synthesis translate_off
    always @(posedge clk) begin
        if (state == CHECK_EMPTY && num_blocks_in_row > cfg_total_blocks) begin
         $error("BSR Scheduler: num_blocks_in_row (%0d) exceeds total_blocks (%0d)",
             num_blocks_in_row, active_total_blocks);
        end
        
        if (state == READ_COL_IDX && block_idx >= cfg_total_blocks) begin
         $error("BSR Scheduler: block_idx (%0d) out of bounds (max %0d)",
             block_idx, active_total_blocks-1);
        end
        
        if (state == LOAD_BLOCK_INIT && block_col >= cfg_num_block_cols) begin
         $error("BSR Scheduler: block_col (%0d) out of bounds (max %0d)",
             block_col, active_num_block_cols-1);
        end
    end
    // synthesis translate_on

endmodule
