# Row-Stationary Dataflow: Complete Weight Flow Trace

## Overview
This document traces EXACTLY how weights move through the system from buffer to PEs, showing the TRUE row-stationary implementation where weights are loaded ONCE and held stationary.

## System Architecture (2x2 array example)

```
         Weight Buffer (wgt_buffer.v)
                   |
              b_vec[15:0]  (TN*8 bits, we use N_COLS=2 so 16 bits)
                   |
            b_in_flat[15:0]
                   |
        +----------+----------+
        |                     |
    b_in[0]              b_in[1]
    (col 0)              (col 1)
        |                     |
    +---+---+             +---+---+
    |       |             |       |
  PE[0,0] PE[1,0]       PE[0,1] PE[1,1]
  (row0)  (row1)        (row0)   (row1)
```

## Phase 1: Weight Loading (S_LOAD_WEIGHT state)

### Scheduler Control Signals
```verilog
// scheduler.v lines 358-371
S_LOAD_WEIGHT: begin
  load_weight = 1'b1;  // Tell PEs to capture weights
  rd_en       = 1'b1;  // Read from weight buffer
  k_idx       = k_ctr; // k_ctr cycles 0,1,2...(Tk_eff-1)
  
  // Cycle through ALL K weights for this tile
  if (k_ctr >= Tk_eff - 1) begin
    state_n = S_STREAM_K;  // Done loading, start streaming
  end
end
```

### Weight Buffer Output
```verilog
// wgt_buffer.v outputs b_vec[TN*8-1:0]
// Example: TN=8, we use first N_COLS elements
// 
// Cycle 0 (k_idx=0): b_vec = [W[7,0], W[6,0], ..., W[1,0], W[0,0]]
//                                                   ↓        ↓
//                                               col1      col0
// Cycle 1 (k_idx=1): b_vec = [W[7,1], W[6,1], ..., W[1,1], W[0,1]]
// Cycle 2 (k_idx=2): b_vec = [W[7,2], W[6,2], ..., W[1,2], W[0,2]]
// ...
// Cycle K-1:         b_vec = [W[7,K-1], W[6,K-1], ..., W[1,K-1], W[0,K-1]]
```

### Systolic Array Distribution
```verilog
// systolic_array.v lines 67-73
// Weight BROADCAST - same weight to all PEs in a column!
for (r = 0; r < N_ROWS; r = r + 1) begin
  for (c = 0; c < N_COLS; c = c + 1) begin
    // BROADCAST weight from b_in (no vertical forwarding)
    wire signed [7:0] b_src = b_in[c];  
    //                        ^^^^^^^^
    // ALL rows in column c get the SAME weight!
  end
end
```

### PE Weight Capture
```verilog
// pe.v lines 76-85
reg signed [7:0] weight_reg;  // STATIONARY storage

always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    weight_reg <= 8'sd0;
  end else begin
    if (load_weight) begin
      weight_reg <= b_in;  // CAPTURE and HOLD
    end
    // Weight STAYS PUT until next load_weight pulse
  end
end
```

### Concrete Example: Loading K=4 weights into 2x2 array

**Cycle 0 (k_idx=0, load_weight=1):**
```
Weight Buffer outputs: b_vec = [..., W[1,0], W[0,0]]
                                        ↓       ↓
Systolic array unpacks: b_in[1]    b_in[0]
                           ↓           ↓
Broadcast to column:    col 1      col 0
                           |           |
                    +------+      +----+
                    ↓      ↓      ↓    ↓
PEs capture:    PE[0,1] PE[1,1] PE[0,0] PE[1,0]
                stores  stores  stores  stores
                W[1,0]  W[1,0]  W[0,0]  W[0,0]
                   ↑       ↑       ↑       ↑
                SAME    SAME    SAME    SAME
                weight  weight  weight  weight
                for all for all for all for all
                in col1 in col1 in col0 in col0
```

**Cycle 1 (k_idx=1, load_weight=1):**
```
Weight Buffer outputs: b_vec = [..., W[1,1], W[0,1]]
                                        ↓       ↓
Broadcast:                          col 1    col 0
                                       |        |
                                   +---+    +---+
                                   ↓   ↓    ↓   ↓
PEs OVERWRITE registers:       PE[0,1] PE[1,1] PE[0,0] PE[1,0]
                               W[1,1]  W[1,1]  W[0,1]  W[0,1]
                               (was    (was    (was    (was
                                W[1,0]) W[1,0]) W[0,0]) W[0,0])
```

**Cycle 2 (k_idx=2, load_weight=1):**
```
PEs now hold: PE[0,1]=W[1,2], PE[1,1]=W[1,2], PE[0,0]=W[0,2], PE[1,0]=W[0,2]
```

**Cycle 3 (k_idx=3, load_weight=1):**
```
PEs now hold: PE[0,1]=W[1,3], PE[1,1]=W[1,3], PE[0,0]=W[0,3], PE[1,0]=W[0,3]
              ^^^^^^^ FINAL   ^^^^^^^ FINAL   ^^^^^^^ FINAL   ^^^^^^^ FINAL
```

**After loading completes:**
- Each PE has captured the LAST weight (k=K-1) from its column
- These weights are now STATIONARY in the weight_reg registers
- Ready for activation streaming phase

## Phase 2: Activation Streaming (S_STREAM_K state)

### Scheduler Control Signals
```verilog
// scheduler.v lines 373-380
S_STREAM_K: begin
  load_weight = 1'b0;  // STOP weight loading (weights stay put)
  
  rd_en = (k_ctr < Tk_eff);  // Read activations
  k_idx = k_ctr;              // Same k indices for activations
  
  en = 1'b1;  // Enable MACs (compute with stationary weights)
  
  // k_ctr cycles 0,1,2...(Tk_eff-1) again for activations
end
```

### PE MAC Operation
```verilog
// pe.v lines 112-116
wire signed [7:0] mac_a = (PIPE) ? a_reg : a_in;  // Activation (FLOWS)
wire signed [7:0] mac_b = weight_reg;             // Weight (STATIONARY!)
//                        ^^^^^^^^^^
//                        Uses stored weight, NOT b_in!

mac8 #(.SAT(SAT)) u_mac (
  .clk(clk), .rst_n(rst_n),
  .a(mac_a),      // Activation changes each cycle
  .b(mac_b),      // Weight CONSTANT (from weight_reg)
  .clr(clr), .en(en),
  .acc(acc)       // Accumulates A*W over K cycles
);
```

### Concrete Example: Streaming K=4 activations

**Initial state (after weight loading):**
```
PE[0,0]: weight_reg = W[0,3]  (STATIONARY)
PE[1,0]: weight_reg = W[0,3]  (STATIONARY - same as PE[0,0])
PE[0,1]: weight_reg = W[1,3]  (STATIONARY)
PE[1,1]: weight_reg = W[1,3]  (STATIONARY - same as PE[0,1])
```

**Cycle 0 (k_idx=0, load_weight=0, en=1):**
```
Activation buffer outputs: a_vec = [A[1,0], A[0,0]]
                                       ↓       ↓
Systolic array receives:           row 1   row 0
                                     |       |
                                     ↓       ↓
MACs compute:                    PE[1,0]  PE[0,0]
                                 A[1,0]*  A[0,0]*
                                 W[0,3]   W[0,3]
                                    ↓        ↓
                              acc[1,0]  acc[0,0]
                                 +=       +=

Activations flow right:          PE[1,0] → PE[1,1]
                                 PE[0,0] → PE[0,1]
                                   
PE[0,1] computes: A[0,0] * W[1,3] → acc[0,1]
PE[1,1] computes: A[1,0] * W[1,3] → acc[1,1]
```

**Cycle 1 (k_idx=1, load_weight=0, en=1):**
```
New activations: a_vec = [A[1,1], A[0,1]]

PE[0,0]: acc += A[0,1] * W[0,3]  (weight_reg UNCHANGED)
PE[1,0]: acc += A[1,1] * W[0,3]  (weight_reg UNCHANGED)
PE[0,1]: acc += A[0,1] * W[1,3]  (weight_reg UNCHANGED)
PE[1,1]: acc += A[1,1] * W[1,3]  (weight_reg UNCHANGED)
```

**Cycles 2-3: Same pattern**
```
Activations change each cycle
Weights STAY PUT in weight_reg
MACs accumulate: acc += a_new * weight_reg
```

## Key Differences from Weight-Flowing Design

### ❌ OLD (Weight-Stationary Claim but Actually Flowing):
```
PE had b_out port → weights flowed vertically
systolic_array had b_fwd[r][c] nets
No load_weight control
Weights updated every cycle from neighbor
```

### ✅ NEW (TRUE Row-Stationary):
```
PE has NO b_out port → weights CANNOT flow
systolic_array broadcasts b_in[c] to ALL rows
load_weight control signal
Weights loaded ONCE in separate phase, then HELD
weight_reg stores weight throughout K cycles
```

## Weight Reuse Efficiency

### Problem Statement
- Matrix multiply: C[M,N] = A[M,K] × B[K,N]
- For each output element C[i,j]: need to multiply K pairs and accumulate

### Row-Stationary Solution
```
Column 0 (PE[*,0]): Load W[0,K-1] → reuse for M activations
Column 1 (PE[*,1]): Load W[1,K-1] → reuse for M activations
...
Column j (PE[*,j]): Load W[j,K-1] → reuse for M activations

REUSE FACTOR: Each weight loaded ONCE, used M times (once per row)
```

### Example for 2x2 array, 8x8 tile (M=8, N=8, K=8)
```
Total weight values needed: K×N = 8×8 = 64 weights
With 2 columns: each weight loaded ONCE, used 8/2 = 4 times
Memory bandwidth: 64 reads (during loading phase)
Compute throughput: 64×4 = 256 MAC ops over streaming phase

Without row-stationary: Would need 64×8 = 512 weight reads!
Bandwidth savings: 8× reduction
```

## Control Timeline

```
TIME →
|
|---- S_PREP_TILE -------|---- S_LOAD_WEIGHT -----|---- S_STREAM_K -----|
     (setup buffers)      (load ALL K weights)     (stream activations)
                          k_ctr: 0→1→2→...→K-1      k_ctr: 0→1→2→...→K-1
                          load_weight=1             load_weight=0
                          PE: weight_reg captures   PE: weight_reg HOLDS
                                                    MAC: acc += a*weight_reg
```

## Summary: Where Do Weights "Flow"?

**They DON'T flow between PEs!**

1. **Weight Buffer → b_vec wire:** Outputs TN*8 bit vector
2. **accel_top slices:** b_in_flat = b_vec[N_COLS*8-1:0]
3. **Systolic array unpacks:** b_in[c] = b_in_flat[c*8 +: 8]
4. **BROADCAST:** All PEs in column c get b_in[c] (no forwarding)
5. **PE captures:** When load_weight=1, weight_reg <= b_in
6. **PE HOLDS:** weight_reg stays constant until next load_weight
7. **MAC uses:** Multiplies a_in (flowing) × weight_reg (stationary)

**The key insight:** 
- Weights "flow" from buffer to PEs during S_LOAD_WEIGHT
- But they DON'T flow BETWEEN PEs (no vertical propagation)
- During S_STREAM_K, weights are 100% stationary in registers
- Activations flow horizontally while weights sit still
- This is TRUE row-stationary dataflow!
