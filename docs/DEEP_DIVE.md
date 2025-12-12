# ACCEL-v1 Architecture Deep Dive

> Technical specification for the 16×16 weight-stationary systolic array accelerator

---

## Table of Contents

1. [Systolic Array Operation](#systolic-array-operation)
2. [Weight-Stationary Dataflow](#weight-stationary-dataflow)
3. [BSR Sparse Scheduling](#bsr-sparse-scheduling)
4. [Timing Analysis](#timing-analysis)
5. [ResNet-18 Layer Breakdown](#resnet-18-layer-breakdown)
6. [Memory Bandwidth Analysis](#memory-bandwidth-analysis)
7. [Power Estimation](#power-estimation)

---

## Systolic Array Operation

### Matrix Multiplication Mapping

For C = A × B where:
- A: Activations [M × K]
- B: Weights [K × N]
- C: Output [M × N]

The 16×16 array computes a 16×16 output tile per pass:

```
                    B (weights)
                    [K × 16]
                    ↓ ↓ ↓ ↓
              ┌─────────────────┐
   A          │                 │
[16 × K] ───▶ │  16×16 Systolic │ ───▶ C [16 × 16]
              │     Array       │
              │                 │
              └─────────────────┘
```

### Tiling for Large Matrices

For M=512, N=512, K=512:

```
Total tiles = ceil(512/16) × ceil(512/16) × ceil(512/16)
            = 32 × 32 × 32
            = 32,768 tile operations

Each tile: 16 × 16 × 16 = 4,096 MACs
Total MACs: 32,768 × 4,096 = 134,217,728 (matches M×N×K)
```

### Tile Loop Structure

```python
# Pseudocode for tiled GEMM
for m_tile in range(0, M, 16):      # Output row tiles
    for n_tile in range(0, N, 16):  # Output col tiles
        # Initialize accumulator tile to 0
        acc[16][16] = 0
        
        for k_tile in range(0, K, 16):  # Reduction tiles
            # Load 16×16 weight block
            load_weights(B[k_tile:k_tile+16, n_tile:n_tile+16])
            
            # Stream 16×16 activation block
            stream_activations(A[m_tile:m_tile+16, k_tile:k_tile+16])
            
            # Accumulate partial products
            acc += systolic_compute()
        
        # Store output tile
        store_output(C[m_tile:m_tile+16, n_tile:n_tile+16], acc)
```

---

## Weight-Stationary Dataflow

### Why Weight-Stationary?

| Dataflow | Weight Reuse | Activation Reuse | Best For |
|----------|--------------|------------------|----------|
| Weight-Stationary | ★★★★★ | ★★☆☆☆ | Large batch, sparse weights |
| Output-Stationary | ★★☆☆☆ | ★★★★☆ | Small batch |
| Row-Stationary | ★★★★☆ | ★★★★☆ | Balanced workloads |

We use **weight-stationary** because:
1. Weights are loaded once per K-tile, reused across all M
2. BSR sparsity means we only load non-zero weight blocks
3. Activation streaming is memory-bound anyway

### Detailed Dataflow Timing

```
Cycle:  0    1    2    3    4    5    6    7   ...   K+14  K+15
        │    │    │    │    │    │    │    │         │     │
PE[0,0]:│ w₀ │ a₀ │ a₁ │ a₂ │ a₃ │ a₄ │ a₅ │ a₆ │...│aₖ₋₁│drain│
PE[0,1]:│ w₁ │    │ a₀ │ a₁ │ a₂ │ a₃ │ a₄ │ a₅ │...│aₖ₋₂│aₖ₋₁│
PE[0,2]:│ w₂ │    │    │ a₀ │ a₁ │ a₂ │ a₃ │ a₄ │...│aₖ₋₃│aₖ₋₂│
   :    │    │    │    │    │    │    │    │    │   │     │
PE[0,15]:│w₁₅│    │    │    │    │    │    │    │...│aₖ₋₁₆│aₖ₋₁₅│
        │    │    │    │    │    │    │    │    │   │     │
        │◄──load──►│◄──────── K cycles compute ─────────►│◄drain►│
        │  (16 cyc) │                                    │(16 cyc)│
```

### PE State Machine

```
        ┌───────────┐
        │   IDLE    │
        └─────┬─────┘
              │ start
              ▼
        ┌───────────┐
        │LOAD_WEIGHT│ ← weight_in valid
        └─────┬─────┘
              │ weight_loaded
              ▼
        ┌───────────┐
        │  COMPUTE  │ ← accumulate: acc += w × a
        └─────┬─────┘
              │ k_done
              ▼
        ┌───────────┐
        │   DRAIN   │ → psum_out valid
        └─────┬─────┘
              │ drained
              ▼
        ┌───────────┐
        │   IDLE    │
        └───────────┘
```

---

## BSR Sparse Scheduling

### Block Skip Logic

The BSR scheduler reads `row_ptr` and `col_idx` to determine which weight blocks to load:

```systemverilog
// Simplified BSR scheduler logic
always_ff @(posedge clk) begin
    case (state)
        IDLE: begin
            if (start) begin
                block_row <= 0;
                block_idx <= row_ptr[0];
                state <= CHECK_ROW;
            end
        end
        
        CHECK_ROW: begin
            if (block_idx < row_ptr[block_row + 1]) begin
                // Non-zero block exists in this row
                current_col <= col_idx[block_idx];
                state <= LOAD_BLOCK;
            end else begin
                // Skip empty row
                block_row <= block_row + 1;
                if (block_row + 1 >= num_block_rows)
                    state <= DONE;
                else
                    block_idx <= row_ptr[block_row + 1];
            end
        end
        
        LOAD_BLOCK: begin
            // DMA 256 bytes from data[block_idx * 256]
            // ... load into weight buffer ...
            block_idx <= block_idx + 1;
            state <= CHECK_ROW;
        end
    endcase
end
```

### Sparsity Speedup Model

```
Dense cycles = M_tiles × N_tiles × K_tiles × (K + 15)
Sparse cycles = M_tiles × N_tiles × nnz_blocks × (K + 15)

Speedup = Dense_cycles / Sparse_cycles
        = total_blocks / nnz_blocks
        = 1 / (1 - sparsity)

Example @ 70% sparsity:
  Speedup = 1 / 0.3 = 3.33×
```

---

## Timing Analysis

### Critical Path

```
Weight BRAM → PE weight reg → Multiplier → Adder → Accumulator reg
   ↓              ↓              ↓           ↓           ↓
  1.2ns        0.3ns          2.5ns       1.5ns       0.5ns = 6.0ns total

Target: 200 MHz (5ns period)
Slack: -1.0ns ❌

Solution: Pipeline the multiplier
  BRAM → reg → MUL stage 1 → MUL stage 2 → ADD → ACC
  1.2    0.3      1.3            1.2        1.5   0.5  = 3.0ns per stage ✓
```

### Latency Breakdown

| Operation | Cycles | Notes |
|-----------|--------|-------|
| CSR config | 10 | AXI-Lite writes |
| DMA BSR header | 50 | 12 bytes @ 64-bit AXI |
| DMA row_ptr | 20 | Per block row |
| DMA col_idx | 5 | Per non-zero block |
| DMA block data | 32 | 256 bytes @ 64-bit |
| Weight load | 16 | Into PE array |
| Compute | K | Stream activations |
| Drain | 16 | Collect partial sums |
| Output write | 32 | 256 bytes @ 64-bit |

### Pipeline Diagram

```
Block 0:  [DMA]──[Load]──[Compute K cycles]──[Drain]
Block 1:         [DMA]──[Load]──[Compute K cycles]──[Drain]
Block 2:                [DMA]──[Load]──[Compute K cycles]──[Drain]
          │      │      │      │
          ◄──────┼──────┼──────► Overlapped: DMA while computing
                 │      │
                 ◄──────► Not overlapped: must wait for load
```

---

## ResNet-18 Layer Breakdown

### Layer Dimensions

| Layer | Type | Input Shape | Weight Shape | Output Shape | MACs |
|-------|------|-------------|--------------|--------------|------|
| conv1 | Conv 7×7, s2 | 3×224×224 | 64×3×7×7 | 64×112×112 | 118M |
| pool | MaxPool 3×3, s2 | 64×112×112 | - | 64×56×56 | - |
| layer1.0.conv1 | Conv 3×3 | 64×56×56 | 64×64×3×3 | 64×56×56 | 231M |
| layer1.0.conv2 | Conv 3×3 | 64×56×56 | 64×64×3×3 | 64×56×56 | 231M |
| layer1.1.conv1 | Conv 3×3 | 64×56×56 | 64×64×3×3 | 64×56×56 | 231M |
| layer1.1.conv2 | Conv 3×3 | 64×56×56 | 64×64×3×3 | 64×56×56 | 231M |
| layer2.0.conv1 | Conv 3×3, s2 | 64×56×56 | 128×64×3×3 | 128×28×28 | 116M |
| layer2.0.conv2 | Conv 3×3 | 128×28×28 | 128×128×3×3 | 128×28×28 | 231M |
| layer2.0.ds | Conv 1×1, s2 | 64×56×56 | 128×64×1×1 | 128×28×28 | 6.4M |
| layer2.1.* | Conv 3×3 ×2 | 128×28×28 | 128×128×3×3 | 128×28×28 | 462M |
| layer3.0.* | Conv 3×3 ×2, s2 | 128×28×28 | 256×...×3×3 | 256×14×14 | 231M |
| layer3.1.* | Conv 3×3 ×2 | 256×14×14 | 256×256×3×3 | 256×14×14 | 231M |
| layer4.0.* | Conv 3×3 ×2, s2 | 256×14×14 | 512×...×3×3 | 512×7×7 | 231M |
| layer4.1.* | Conv 3×3 ×2 | 512×7×7 | 512×512×3×3 | 512×7×7 | 231M |
| avgpool | Global | 512×7×7 | - | 512×1×1 | - |
| fc | Linear | 512 | 1000×512 | 1000 | 0.5M |
| **TOTAL** | | | | | **1.82B** |

### im2col Transformation for Conv

Convolutions are mapped to GEMM via im2col:

```
Conv2D: Input [C_in × H × W] * Kernel [C_out × C_in × kH × kW]
      ↓ im2col
GEMM:   A [H_out×W_out × C_in×kH×kW] × B [C_in×kH×kW × C_out]
      = C [H_out×W_out × C_out]

Example: layer1.0.conv1
  Input: 64×56×56, Kernel: 64×64×3×3
  im2col A: [3136 × 576]  (3136 = 56×56, 576 = 64×3×3)
  Weight B: [576 × 64]
  Output C: [3136 × 64]
  
  Tiles: ceil(3136/16) × ceil(64/16) × ceil(576/16)
       = 196 × 4 × 36 = 28,224 tile ops
```

### Cycle Estimates per Layer

Assuming 200 MHz, 70% block sparsity:

| Layer | Tiles (Dense) | Tiles (Sparse) | Cycles | Time (ms) |
|-------|---------------|----------------|--------|-----------|
| conv1 | 784 × 4 × 10 = 31K | 9.4K | 160K | 0.80 |
| layer1.* (4 conv) | 196 × 4 × 36 × 4 = 113K | 34K | 578K | 2.89 |
| layer2.* (5 conv) | ~80K | 24K | 408K | 2.04 |
| layer3.* (5 conv) | ~40K | 12K | 204K | 1.02 |
| layer4.* (5 conv) | ~20K | 6K | 102K | 0.51 |
| fc | 63 × 1 × 32 = 2K | 0.6K | 10K | 0.05 |
| **TOTAL** | ~286K | ~86K | **1.46M** | **7.3 ms** |

**Throughput: ~137 images/second @ 70% sparsity**

---

## Memory Bandwidth Analysis

### DDR Requirements

```
Per image:
  Weights: 11.7 MB (FP32) → 2.9 MB (INT8) → ~0.9 MB (70% sparse BSR)
  Activations: ~2 MB peak (largest feature map)
  Outputs: ~0.5 MB (before pooling)
  
Bandwidth @ 137 img/s:
  Weights: 0.9 MB × 137 = 123 MB/s (fits in cache after first image)
  Activations: 2 MB × 137 = 274 MB/s
  Outputs: 0.5 MB × 137 = 68 MB/s
  
Total: ~465 MB/s << Z7020's 4.2 GB/s DDR bandwidth ✓
```

### On-Chip Buffer Sizing

| Buffer | Size | Purpose |
|--------|------|---------|
| Weight BRAM | 16 KB | Current 16×16 block + next (double buffer) |
| Activation BRAM | 8 KB | 16 rows × 256 cols × 2 banks |
| Output BRAM | 4 KB | 16×16 × INT32 × 2 banks |
| BSR Metadata | 2 KB | row_ptr, col_idx for current layer |
| **Total** | **30 KB** | Fits in Z7020's 560 KB BRAM ✓ |

---

## Power Estimation

### Component Breakdown (@ 200 MHz, Z7020)

| Component | Count | Power/Unit | Total |
|-----------|-------|------------|-------|
| DSP48 (MAC) | 196 | 5 mW | 980 mW |
| BRAM | 30 | 10 mW | 300 mW |
| Logic (LUTs) | 18K | 0.01 mW | 180 mW |
| Registers | 12K | 0.005 mW | 60 mW |
| Clocking | - | - | 100 mW |
| I/O (AXI) | - | - | 50 mW |
| **Static** | - | - | 200 mW |
| **Total** | | | **1.87 W** |

### Energy Efficiency

```
Throughput: 137 img/s
Power: 1.87 W
Energy: 1.87 / 137 = 13.6 mJ/image

Comparison:
  CPU (i7): ~1000 mJ/image
  GPU (RTX 3090): ~50 mJ/image
  ACCEL-v1 (Z7020): ~14 mJ/image ← 3.5× better than GPU
```

### Power Optimization Techniques (Implemented)

1. **Clock gating**: Disable PE clocks when idle
2. **Block skipping**: Don't clock zero-weight blocks
3. **Activation gating**: Gate datapath for zero activations
4. **BRAM power-down**: Disable unused banks

---

## Summary: Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Array size | 16×16 (or 14×14 for Z7020) | 256 PEs |
| Block size | 16×16 | Matches array |
| Data type | INT8 weights, INT8 activations | INT32 accumulators |
| Clock | 200 MHz target | 5ns period |
| Throughput | 51.2 GOPS (dense) | 256 MACs × 200 MHz |
| Throughput | 170 GOPS (70% sparse) | 3.3× speedup |
| Latency | 7.3 ms/image | ResNet-18 @ 70% sparse |
| Power | ~1.9 W | Z7020 @ 200 MHz |
| Efficiency | 90 GOPS/W | At 70% sparsity |
