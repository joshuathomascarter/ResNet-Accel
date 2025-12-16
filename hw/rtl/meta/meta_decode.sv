// =============================================================================
// meta_decode.sv — BSR Metadata Decoder with Direct-Mapped Cache
// =============================================================================
//
// DESCRIPTION:
// ------------
// Provides cached access to BSR metadata (row pointers and column indices)
// stored in BRAM. Reduces effective memory latency for frequently accessed
// entries by caching recent lookups.
//
// The scheduler frequently reads row_ptr[i] and row_ptr[i+1] to determine
// the range of non-zero blocks in each row. Without caching, each access
// would incur BRAM latency. With caching, sequential row accesses hit cache.
//
// CACHE ARCHITECTURE:
// -------------------
//   Direct-mapped cache with CACHE_DEPTH entries:
//     - Index: addr[5:0] (64-entry cache)
//     - Tag: None (index-only lookup, valid bit only)
//     - Data: 32-bit (matches row_ptr width)
//
//   This simple design trades some hit rate for low latency:
//     - No tag comparison needed (just valid bit check)
//     - Single-cycle hit path
//     - 2-cycle miss path (BRAM read latency)
//
// TIMING:
// -------
//   Cache Hit:  req_valid → meta_valid in 2 cycles
//               IDLE → CHECK_CACHE → OUTPUT_DATA
//
//   Cache Miss: req_valid → meta_valid in 3 cycles
//               IDLE → READ_META → WAIT_MEM → OUTPUT_DATA
//
//   Pipeline optimization: If a new request arrives while OUTPUT_DATA,
//   skip IDLE state and proceed directly to next lookup.
//
// FSM STATE DIAGRAM:
// ------------------
//   ┌───────────────────────────────────────────┐
//   │                                           │
//   │    ┌─────┐    hit    ┌─────────────┐     │
//   └───►│IDLE │──────────►│CHECK_CACHE  │─────┤
//        └──┬──┘           └──────┬──────┘     │
//           │ miss                │            │
//           ▼                     │            │
//        ┌──────────┐             │            │
//        │READ_META │             │            │
//        └────┬─────┘             │            │
//             │                   │            │
//             ▼                   │            │
//        ┌──────────┐             │            │
//        │WAIT_MEM  │             │            │
//        └────┬─────┘             │            │
//             │                   │            │
//             └─────────┬─────────┘            │
//                       │                      │
//                       ▼                      │
//                 ┌───────────┐                │
//                 │OUTPUT_DATA│────────────────┘
//                 └───────────┘  (meta_ready && req_valid → pipeline)
//
// LIMITATIONS:
// ------------
// - No tag storage: Aliased entries will cause incorrect data on conflict
//   This is acceptable since scheduler accesses row_ptr sequentially
// - 64-entry cache covers 64 row pointers (64 block rows = 896 matrix rows)
//   Sufficient for typical layer sizes
// - Valid bits cleared only on reset (no replacement/invalidation policy)
//
// RESOURCE ESTIMATES:
// -------------------
//   - BRAM: 64 × 32 bits = 256 bytes (fits in distributed RAM)
//   - FFs: ~40 (FSM state, latches, valid bits)
//   - LUTs: ~80 (FSM logic, muxes)
//
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module meta_decode #(
    // =========================================================================
    // PARAMETERS
    // =========================================================================
    // DATA_WIDTH: Metadata entry width (32-bit for row pointers)
    // - Matches row_ptr format: 32-bit unsigned integer
    // - Could be 16-bit for col_idx, but 32-bit is more general
    parameter DATA_WIDTH = 32,
    
    // CACHE_DEPTH: Number of cache entries
    // - 64 entries × 32 bits = 256 bytes (fits in distributed RAM)
    // - Index uses addr[5:0], so accesses modulo 64 map to same entry
    // - For typical layers (M ≤ 512), 64 entries covers 64 block rows
    parameter CACHE_DEPTH = 64
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // =========================================================================
    // SCHEDULER REQUEST INTERFACE
    // =========================================================================
    // Scheduler requests metadata by address (row_ptr index × 4 for byte addr).
    // req_valid/req_ready handshake follows AXI-Stream convention.
    input  wire                     req_valid,      // Request pending
    input  wire [31:0]              req_addr,       // Metadata address (byte)
    output reg                      req_ready,      // Can accept request

    // =========================================================================
    // MEMORY INTERFACE (to row_ptr BRAM)
    // =========================================================================
    // Direct connection to row_ptr_bram in accel_top.
    // Single-cycle read latency (data valid cycle after mem_en).
    output wire                     mem_en,         // Read enable
    output wire [31:0]              mem_addr,       // Read address
    input  wire [DATA_WIDTH-1:0]    mem_rdata,      // Read data (1-cycle latency)

    // =========================================================================
    // SCHEDULER RESPONSE INTERFACE
    // =========================================================================
    // Returns fetched metadata to scheduler.
    // meta_valid/meta_ready handshake follows AXI-Stream convention.
    output reg                      meta_valid,     // Data ready
    output reg [DATA_WIDTH-1:0]     meta_rdata,     // Metadata value
    input  wire                     meta_ready      // Scheduler accepting
);

    // =========================================================================
    // FSM STATE ENCODING (One-Hot)
    // =========================================================================
    // One-hot encoding for faster state decoding and better timing.
    // fsm_encoding attribute hints to synthesis tool.
    localparam [5:0] S_IDLE         = 6'b000001,  // Waiting for request
                     S_READ_META    = 6'b000010,  // Initiate BRAM read (miss)
                     S_WAIT_MEM     = 6'b000100,  // Wait for BRAM data
                     S_CHECK_CACHE  = 6'b001000,  // Read from cache (hit)
                     S_OUTPUT_DATA  = 6'b010000,  // Present data to scheduler
                     S_DONE         = 6'b100000;  // Unused (legacy)

    (* fsm_encoding = "one_hot" *) reg [5:0] current_state, next_state;

    // =========================================================================
    // CACHE DATA STRUCTURES
    // =========================================================================
    // cache_mem: Direct-mapped cache storage
    //   - CACHE_DEPTH entries × DATA_WIDTH bits
    //   - Indexed by addr[5:0] for 64-entry cache
    reg [DATA_WIDTH-1:0] cache_mem [0:CACHE_DEPTH-1];
    
    // cache_valid_bits: One bit per cache entry
    //   - 1 = entry contains valid data
    //   - 0 = entry is empty/invalid
    //   - Cleared on reset, set on BRAM fetch
    reg [CACHE_DEPTH-1:0] cache_valid_bits;
    
    // addr_latch: Captured request address
    //   - Held constant during multi-cycle lookup
    reg [31:0] addr_latch;
    
    // cache_index: Current lookup index (lower 6 bits of address)
    wire [5:0] cache_index;
    
    // fetched_data: Data from BRAM read (used on miss path)
    reg [DATA_WIDTH-1:0] fetched_data;

    // =========================================================================
    // ADDRESS LATCHING
    // =========================================================================
    // Capture request address on handshake for use during multi-cycle lookup.
    // This allows req_addr to change while we're still processing.
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            addr_latch <= 32'd0;
        end else if (req_valid && req_ready) begin
            addr_latch <= req_addr;
        end
    end

    // active_addr: Currently processing address
    // Use direct req_addr during handshake for 1-cycle lookahead,
    // otherwise use latched address.
    wire [31:0] active_addr = (req_ready && req_valid) ? req_addr : addr_latch;

    //-------------------------------------------------------------------------
    // FSM STATE REGISTER
    //-------------------------------------------------------------------------

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state <= S_IDLE;
        end else begin
            current_state <= next_state;
        end
    end

    //-------------------------------------------------------------------------
    // FSM NEXT STATE LOGIC
    //-------------------------------------------------------------------------
    // Combinational logic determines next state based on current state and
    // input conditions. Pipeline optimization skips IDLE when new request
    // arrives during OUTPUT_DATA.
    
    always @(*) begin
        next_state = current_state;
        
        case (current_state)
            // S_IDLE: Wait for scheduler request
            S_IDLE: begin
                if (req_valid) begin
                    // Lookahead: Check hit/miss on the incoming address immediately
                    if (cache_valid_bits[req_addr[5:0]]) begin
                        next_state = S_CHECK_CACHE; // Hit: 2-cycle path
                    end else begin
                        next_state = S_READ_META;   // Miss: 3-cycle path
                    end
                end
            end

            // S_READ_META: Initiate BRAM read (cache miss)
            // mem_en asserted, BRAM latches address
            S_READ_META: begin
                next_state = S_WAIT_MEM;
            end

            // S_WAIT_MEM: Wait for BRAM data (1-cycle latency)
            // Data valid at end of this cycle, captured in registers
            S_WAIT_MEM: begin
                next_state = S_OUTPUT_DATA;
            end

            // S_CHECK_CACHE: Cache hit path
            // Data already in cache_mem, proceed to output
            S_CHECK_CACHE: begin
                next_state = S_OUTPUT_DATA;
            end

            // S_OUTPUT_DATA: Present data to scheduler
            // Pipeline optimization: Skip IDLE if new request waiting
            S_OUTPUT_DATA: begin
                if (meta_ready) begin
                    if (req_valid) begin
                        // New request waiting - pipeline directly
                        if (cache_valid_bits[req_addr[5:0]]) 
                            next_state = S_CHECK_CACHE;
                        else 
                            next_state = S_READ_META;
                    end else begin
                        next_state = S_IDLE;
                    end
                end
            end

            // S_DONE: Unused legacy state
            S_DONE: begin
                next_state = S_IDLE;
            end
            
            default: next_state = S_IDLE;
        endcase
    end

    //-------------------------------------------------------------------------
    // DATAPATH: Cache Index and Memory Interface
    //-------------------------------------------------------------------------

    // Cache index: Lower 6 bits of address (modulo 64)
    assign cache_index = active_addr[5:0];

    // Memory interface: Enable only on cache miss
    assign mem_en   = (current_state == S_READ_META);
    assign mem_addr = active_addr;

    //-------------------------------------------------------------------------
    // DATAPATH: Cache Write and Valid Bit Update
    //-------------------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cache_valid_bits <= {CACHE_DEPTH{1'b0}};
            fetched_data <= 32'd0;
        end else if (current_state == S_WAIT_MEM) begin
            // BRAM data available: Store in cache and fetched_data
            cache_mem[cache_index] <= mem_rdata;
            cache_valid_bits[cache_index] <= 1'b1;
            fetched_data <= mem_rdata;
        end
    end

    //-------------------------------------------------------------------------
    // OUTPUT LOGIC
    //-------------------------------------------------------------------------
    always @(*) begin
        // req_ready: Can accept if IDLE or finishing current request
        req_ready  = (current_state == S_IDLE) || (current_state == S_OUTPUT_DATA && meta_ready);
        
        // meta_valid: Data ready for scheduler
        meta_valid = (current_state == S_OUTPUT_DATA);
        
        // meta_rdata: Mux between cache hit and BRAM fetch paths
        if (current_state == S_OUTPUT_DATA) begin
             if (cache_valid_bits[cache_index]) 
                meta_rdata = cache_mem[cache_index];
             else 
                meta_rdata = fetched_data; 
        end else begin
             meta_rdata = 32'd0;
        end
    end

endmodule

`default_nettype wire
