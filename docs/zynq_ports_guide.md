# Understanding Zynq-7000 Ports for ACCEL-v1

**Simple Guide: How PS and PL Talk to Each Other**

---

## Quick Visual: Your Project's Communication

```
┌────────────────────────────────────────────────────┐
│              ARM PS (Running PYNQ Python)          │
│                                                     │
│  Python code:                                      │
│    accel.write(0x00, 0x1)  # START                │
│                    ↓                               │
│                  GP0 Port                          │
│            (AXI4-Lite, 32-bit)                     │
│              For control only                      │
│                    ↓                               │
├────────────────────┼────────────────────────────────┤
│              DDR Memory (512 MB)                   │
│    [weights] [activations] [results]              │
│         ↑                    ↓                     │
│       HP0 Port          HP0 Port                   │
│   (AXI4, 64-bit)    (AXI4, 64-bit)                │
│    Read data        Write results                  │
│         ↑                    ↓                     │
└─────────┼────────────────────┼─────────────────────┘
          │                    │
          │                    │
┌─────────┼────────────────────┼─────────────────────┐
│  FPGA PL│                    │                     │
│         │                    ▼                     │
│    ┌────┴─────────┐    ┌────────────┐            │
│    │ AXI Master   │    │ AXI Lite   │            │
│    │ (DMA Bridge) │    │ Slave(CSRs)│            │
│    └────┬─────────┘    └────────────┘            │
│         │                                          │
│         ▼                                          │
│   [Your 8×8 Systolic Array]                       │
└──────────────────────────────────────────────────┘
```

---

## The 3 Main Port Types (Explained Simply)

### 1. GP0 Port (General Purpose) - "The Remote Control"

**What it is:**
- Slow, narrow interface (32-bit)
- Used for control and status
- Low bandwidth (~400 MB/s, but you use <1 MB/s)

**What YOU use it for:**
- Python writes to your CSR registers
- Starting/stopping the accelerator
- Checking status (DONE, BUSY, ERROR)
- Configuring addresses

**Example Python code:**
```python
# These all go through GP0:
accel.write(0x00, 0x1)              # Write START bit
status = accel.read(0x04)           # Read STATUS register
accel.write(0x08, weights_addr)     # Configure weight address
```

**In your Verilog:**
```verilog
// This is your AXI4-Lite Slave connected to GP0
module axi_lite_slave (
    input [11:0] s_axi_awaddr,   // ← Comes from GP0
    input [31:0] s_axi_wdata,    // ← Python's write data
    output [31:0] s_axi_rdata,   // → Status back to Python
    ...
);
```

**Think of it as:** Your accelerator's "remote control" - buttons and status LEDs.

---

### 2. HP0 Port (High Performance) - "The Data Highway"

**What it is:**
- Fast, wide interface (64-bit)
- Used for bulk data transfer
- High bandwidth (~1200 MB/s available)

**What YOU use it for:**
- Reading weights from DDR (106 MB/s)
- Reading activations from DDR
- Writing results back to DDR (337 MB/s)
- Total: ~443 MB/s (37% of HP0 capacity)

**How data flows:**
```
Python allocates buffer in DDR:
  weights_buf = allocate(xlnk, ...)
  weights_buf.physical_address = 0x1000_0000

Python tells your accelerator where data is:
  accel.write(0x08, 0x1000_0000)  ← Goes via GP0

Your AXI Master reads from that address:
  AXI Master → HP0 → DDR @ 0x1000_0000 → Data → Your accelerator
```

**In your Verilog:**
```verilog
// This is your AXI4 Master connected to HP0
module axi_dma_bridge (
    output [31:0] m_axi_araddr,   // → Read address to HP0
    input [63:0] m_axi_rdata,     // ← 64-bit data from DDR
    output [31:0] m_axi_awaddr,   // → Write address to HP0
    output [63:0] m_axi_wdata,    // → 64-bit data to DDR
    ...
);
```

**Think of it as:** A highway for moving large amounts of data (weights, activations, results).

---

### 3. ACP Port (Accelerator Coherency) - "Cache Access"

**What it is:**
- Cache-coherent interface to L1/L2 cache
- Lower latency for small data
- Good for CPU-PL shared data structures

**What YOU use it for:**
- **NOTHING!** Don't use it.

**Why NOT use it:**
- Your matrices are HUGE (MB scale)
- L2 cache is only 512 KB
- Weights don't fit in cache
- Cache thrashing would make it slower
- HP0 is better for bulk transfers

**When you WOULD use ACP:**
- Small lookup tables (<100 KB)
- Frequent CPU-PL data sharing
- Latency-critical small data
- **NOT your case!**

**Think of it as:** A fast lane for small packages. You're shipping furniture (matrices), so use the truck (HP0) instead.

---

## Complete Data Flow Example

Let's trace what happens when you run your accelerator:

### Step 1: Python Setup (via GP0)
```python
# Allocate DDR buffers
weights = allocate(xlnk, shape=(7316,), dtype=np.int8)      # 7.3 KB
activations = allocate(xlnk, shape=(128,), dtype=np.int8)
results = allocate(xlnk, shape=(1024,), dtype=np.int32)

# Configure accelerator (all via GP0)
accel.write(0x08, weights.physical_address)      # ← GP0
accel.write(0x18, activations.physical_address)  # ← GP0
accel.write(0x20, results.physical_address)      # ← GP0
accel.write(0x28, 128)  # num_rows                # ← GP0
accel.write(0x2C, 784)  # num_cols                # ← GP0
accel.write(0x00, 0x1)  # START!                  # ← GP0
```

### Step 2: Your Accelerator Runs (via HP0)
```
1. AXI Lite Slave receives START via GP0
   └─> Triggers sparse_controller FSM

2. Sparse Controller requests metadata:
   └─> AXI Master reads row_ptr via HP0 (from DDR)
   └─> AXI Master reads col_idx via HP0 (from DDR)

3. For each non-zero block:
   └─> AXI Master reads 8×8 block via HP0 (64 bytes)
   └─> AXI Master reads activations via HP0 (8 bytes)
   └─> Systolic array computes
   └─> Results → Output FIFO

4. Output buffer full:
   └─> AXI Master writes results via HP0 (to DDR)

5. Done:
   └─> AXI Lite Slave sets DONE bit in STATUS register
```

### Step 3: Python Checks Completion (via GP0)
```python
# Poll STATUS register via GP0
while (accel.read(0x04) & 0x1) == 0:  # ← GP0
    time.sleep(0.001)

print("Done!")
print("Results at DDR address:", results.physical_address)
print("Results:", np.array(results))  # NumPy reads from DDR
```

---

## Bandwidth Summary for Your Project

| Interface | You Use? | For What?              | Your Usage   | Max Available |
|-----------|----------|------------------------|--------------|---------------|
| **GP0**   | ✅ YES   | Control (CSRs)         | ~1 KB        | 400 MB/s      |
| **HP0**   | ✅ YES   | Data (weights/results) | ~443 MB/s    | 1200 MB/s     |
| **ACP**   | ❌ NO    | Not needed             | 0            | 600 MB/s      |

**Your bottleneck:** Computation (64 MACs/cycle), NOT bandwidth!

---

## Why Each Port Has Its Job

**GP0 is slow but simple:**
- Only needs to send a few bytes (register addresses, values)
- Python → GP0 → CSRs is low latency for control
- Perfect for "start", "stop", "status" operations

**HP0 is fast and wide:**
- Needs to move MB/s of data continuously
- Burst transfers (read 16×64-bit words at once)
- Optimized for streaming weights/activations

**ACP is cache-coherent:**
- ARM can write to cache, PL sees it instantly
- Good for shared data structures
- BAD for large matrices (cache too small)

---

## What You Actually Implement

### In your Verilog top module:
```verilog
module accel_top (
    // GP0 connection (control)
    input [11:0] s_axi_awaddr,
    input [31:0] s_axi_wdata,
    output [31:0] s_axi_rdata,
    // ... rest of AXI4-Lite signals

    // HP0 connection (data)
    output [31:0] m_axi_araddr,    // Read address
    input [63:0] m_axi_rdata,      // Read data (64-bit!)
    output [31:0] m_axi_awaddr,    // Write address
    output [63:0] m_axi_wdata,     // Write data (64-bit!)
    // ... rest of AXI4 Master signals

    // Clock and reset from PS
    input aclk,
    input aresetn
);
```

### In your Vivado block design:
```
[ZYNQ PS] ──GP0──> [Your AXI Lite Slave] ──> [CSRs]
    │
    └──HP0──> [Your AXI Master] ──> [Systolic Array]
         ↑
         └─ DDR Memory
```

---

## Common Mistakes to Avoid

**❌ DON'T:**
- Use ACP for large matrices (too big for cache)
- Use GP0 for data transfer (too slow)
- Try to access DDR directly without AXI (not possible)

**✅ DO:**
- Use GP0 for control only (START, STATUS, addresses)
- Use HP0 for all data movement (weights, activations, results)
- Use 64-bit bursts on HP0 (faster than single reads)

---

## Summary: What This All Means for You

**Your project uses 2 ports:**

1. **GP0 (AXI4-Lite):** Control interface
   - Python configures your accelerator
   - Python starts/stops computation
   - Python reads performance counters

2. **HP0 (AXI4):** Data interface
   - Your AXI Master reads weights from DDR
   - Your AXI Master writes results to DDR
   - Fast bulk transfers (443 MB/s)

**You DON'T use:**
- ACP (cache too small for matrices)
- Other HP ports (HP0 is enough)
- Interrupts (polling STATUS is simpler for now)

**Your bandwidth is fine:**
- You need 443 MB/s
- HP0 provides 1200 MB/s
- 37% utilization = plenty of headroom

---

## Next Steps for Nov 24

Now that you understand the ports, complete your architecture spec:

1. ✅ Data path diagram (showing GP0 and HP0)
2. ✅ Module hierarchy (AXI Lite Slave + AXI Master)
3. ✅ Interface specs (all signal widths documented)
4. ✅ Bandwidth calculations (443 MB/s < 1200 MB/s)
5. ✅ Performance model (6.4 GOPS dense, 54 GOPS sparse)

**You're ready to start coding on Nov 26!**
