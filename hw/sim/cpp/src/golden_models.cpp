/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                         GOLDEN_MODELS.CPP                                 ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  File        : golden_models.cpp                                          ║
 * ║  Description : Bit-exact reference implementations matching RTL hardware  ║
 * ║  Author      : ResNet-Accel Team                                          ║
 * ║  Date        : 2024                                                       ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║    These "golden models" produce the EXACT same results as hardware.      ║
 * ║    They are used for verification: run the same input through both        ║
 * ║    hardware (RTL simulation) and these C++ models, compare outputs.       ║
 * ║    If they don't match bit-for-bit, there's a bug somewhere.              ║
 * ║                                                                           ║
 * ║  KEY INSIGHT - WHY BIT-EXACT MATTERS:                                     ║
 * ║    INT8 quantized inference is very sensitive to rounding. A single       ║
 * ║    bit difference in one layer can cascade through the network.           ║
 * ║    Golden models must use:                                                ║
 * ║      • Same data types (int8_t, int32_t) as hardware                     ║
 * ║      • Same accumulation order as the systolic array dataflow            ║
 * ║      • Same rounding mode (round-half-to-even / banker's rounding)       ║
 * ║      • Same saturation behavior (clamp, don't wrap)                      ║
 * ║                                                                           ║
 * ║  ARCHITECTURE REFERENCE (14×14 Weight-Stationary Systolic Array):         ║
 * ║                                                                           ║
 * ║    GEMM: C[M×N] = A[M×K] × B[K×N]                                         ║
 * ║                                                                           ║
 * ║    ┌─────────────────────────────────────────────────────────────────┐    ║
 * ║    │                  Activations A (flow →)                         │    ║
 * ║    │               k=0   k=1   k=2  ...  k=K-1                       │    ║
 * ║    │    ┌─────┬─────┬─────┬─────┬─────┐                              │    ║
 * ║    │ m=0│ PE  │ PE  │ PE  │ ... │ PE  │ → acc → C[0][n]              │    ║
 * ║    │    ├─────┼─────┼─────┼─────┼─────┤                              │    ║
 * ║    │ m=1│ PE  │ PE  │ PE  │ ... │ PE  │ → acc → C[1][n]              │    ║
 * ║    │    ├─────┼─────┼─────┼─────┼─────┤                              │    ║
 * ║    │    │ ... │ ... │ ... │ ... │ ... │                              │    ║
 * ║    │    ├─────┼─────┼─────┼─────┼─────┤                              │    ║
 * ║    │m=13│ PE  │ PE  │ PE  │ ... │ PE  │ → acc → C[13][n]             │    ║
 * ║    │    └─────┴─────┴─────┴─────┴─────┘                              │    ║
 * ║    │                  ↑                                              │    ║
 * ║    │          Weights B (stationary in PEs)                          │    ║
 * ║    │                                                                 │    ║
 * ║    │    Each PE: acc += A[m][k] × B[k][n]                            │    ║
 * ║    │    After K cycles: C[m][n] = Σ(A[m][k] × B[k][n]) for k=0..K-1  │    ║
 * ║    └─────────────────────────────────────────────────────────────────┘    ║
 * ║                                                                           ║
 * ║  BSR SPARSE FORMAT:                                                       ║
 * ║    Block Sparse Row format stores only non-zero 14×14 blocks.             ║
 * ║    Block size = 14 matches systolic array dimensions.                     ║
 * ║                                                                           ║
 * ║    row_ptr[i]: Start index of blocks in row i                            ║
 * ║    col_idx[j]: Column index of block j                                   ║
 * ║    data[j*196..(j+1)*196-1]: INT8 values for block j (14×14 = 196)       ║
 * ║                                                                           ║
 * ║    Example (2 block-rows, 3 non-zero blocks):                            ║
 * ║    row_ptr = [0, 2, 3]       # Row 0 has 2 blocks, row 1 has 1           ║
 * ║    col_idx = [0, 2, 1]       # Blocks at (0,0), (0,2), (1,1)             ║
 * ║    data    = [196 vals][196 vals][196 vals]                              ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include "golden_models.hpp"
#include "bsr_packer.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cfenv>
#include <vector>

namespace resnet_accel {
namespace golden {

//==============================================================================
// DENSE MATRIX MULTIPLICATION (GEMM)
//==============================================================================
/**
 * Dense Matrix Multiply: C[M×N] = A[M×K] × B[K×N]
 *
 * This is the reference implementation for dense (non-sparse) GEMM.
 * Used for layers with <50% sparsity where BSR format isn't efficient.
 *
 * Data Types:
 *   A, B: INT8 (signed, range [-128, +127])
 *   C:    INT32 (accumulator, prevents overflow)
 *
 * Accumulator Sizing Math:
 *   Worst case per multiply: (-128) × (-128) = 16,384
 *   Maximum K dimension: ~4096 (for ResNet-18 FC layers)
 *   Worst case sum: 4096 × 16,384 = 67,108,864 (fits in INT32: ±2.1B)
 *
 * Loop Order: M → N → K (output-stationary)
 *   This matches how we read results from the systolic array:
 *   - Complete one output element before moving to the next
 *   - K loop is the reduction dimension (innermost)
 *
 * Memory Layout: Row-major (C-style)
 *   A[m][k] = A[m * K + k]
 *   B[k][n] = B[k * N + n]
 *   C[m][n] = C[m * N + n]
 *
 * @param A  Activation matrix [M × K], row-major, INT8
 * @param B  Weight matrix [K × N], row-major, INT8
 * @param C  Output matrix [M × N], row-major, INT32 (caller must allocate)
 * @param M  Number of output rows (batch × spatial_size for conv)
 * @param K  Reduction dimension (inner product length)
 * @param N  Number of output columns (output channels)
 */
void matmul_int8(const std::int8_t* A, const std::int8_t* B, std::int32_t* C,
                 std::size_t M, std::size_t K, std::size_t N) {
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            // Initialize accumulator to zero
            // Hardware: 32-bit accumulator register in each PE
            std::int32_t acc = 0;
            
            // Inner product: sum of element-wise products
            // Hardware: This loop executes over K clock cycles
            for (std::size_t k = 0; k < K; ++k) {
                // Explicit cast to INT32 BEFORE multiply to prevent overflow
                // Hardware: 8×8→16 multiplier, extended to 32-bit for accumulation
                acc += static_cast<std::int32_t>(A[m * K + k]) * 
                       static_cast<std::int32_t>(B[k * N + n]);
            }
            
            // Store result (no saturation needed - INT32 won't overflow for reasonable K)
            C[m * N + n] = acc;
        }
    }
}

//==============================================================================
// BSR SPARSE MATRIX MULTIPLICATION
//==============================================================================
/**
 * Sparse Matrix Multiply using BSR Format: C[M×N] = A[M×K] × B_bsr[K×N]
 *
 * THIS IS THE KEY FUNCTION - it must match hardware bit-exactly!
 *
 * BSR (Block Sparse Row) Format:
 *   Instead of storing individual non-zero elements, BSR stores non-zero
 *   BLOCKS. For our 14×14 systolic array, block_size = 14.
 *
 *   Benefits:
 *   - One block fills entire array (perfect utilization)
 *   - Reduced metadata overhead vs CSR
 *   - Coalesced memory access (196 bytes per block)
 *
 *   Structure:
 *   - row_ptr[num_block_rows + 1]: Start index of blocks for each block-row
 *   - col_idx[nnz_blocks]: Column index for each non-zero block
 *   - data[nnz_blocks × 196]: Flattened INT8 values (14×14 per block)
 *
 * Algorithm Walkthrough:
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │  For each block-row br in B_bsr (0 to num_block_rows-1):            │
 *   │    For each non-zero block at column bc:                            │
 *   │      block_data = 14×14 INT8 values                                 │
 *   │      For each row m in A:                                           │
 *   │        For each column j in block (0..13):                          │
 *   │          n = bc * 14 + j   (actual column in output)                │
 *   │          For each row i in block (0..13):                           │
 *   │            k = br * 14 + i   (actual row in K dimension)            │
 *   │            C[m][n] += A[m][k] * block[i][j]                         │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 * Hardware Mapping:
 *   - Each block fills the weight buffer (14×14 weights stationary)
 *   - Activations stream from act_buffer (14 elements per cycle)
 *   - 14 outputs computed in parallel (one column of C)
 *   - After 14 cycles, one 14×14 tile of C is complete
 *
 * Memory Access Pattern:
 *   - A: Sequential row access (good cache behavior)
 *   - B_bsr.data: Block-sequential (excellent cache behavior)
 *   - C: Random column access (may have cache misses for large N)
 *
 * @param A      Dense activation matrix [M × K], row-major, INT8
 * @param B_bsr  Sparse weight matrix in BSR format
 * @param C      Output matrix [M × N], row-major, INT32 (must be pre-zeroed!)
 * @param M      Number of output rows
 * @param K      Reduction dimension (should equal B_bsr.num_block_rows × 14)
 * @param N      Number of output columns (should equal B_bsr.num_block_cols × 14)
 */
void bsr_matmul_int8(const std::int8_t* A, const BSRMatrix& B_bsr, std::int32_t* C,
                     std::size_t M, std::size_t K, std::size_t N) {
    // CRITICAL: Zero-initialize output matrix before accumulation
    // Hardware: Output accumulator is reset to 0 at start of each tile
    std::memset(C, 0, M * N * sizeof(std::int32_t));
    
    // Iterate over block rows of B (each block-row corresponds to 14 rows of K)
    // Hardware: bsr_scheduler iterates through row_ptr to find blocks
    for (std::size_t br = 0; br < B_bsr.num_block_rows; ++br) {
        
        // row_ptr[br] to row_ptr[br+1] gives the range of non-zero blocks in this row
        // Hardware: meta_decode caches row_ptr for fast lookup
        for (std::size_t idx = B_bsr.row_ptr[br]; idx < B_bsr.row_ptr[br + 1]; ++idx) {
            
            // col_idx[idx] tells us which block-column this non-zero block is in
            // Hardware: col_idx fetched from DDR via bsr_dma
            std::size_t bc = B_bsr.col_idx[idx];
            
            // Pointer to this block's 196 INT8 weight values (14×14 = 196)
            // Hardware: These 196 bytes are loaded into wgt_buffer
            // Block layout: block[i * 14 + j] = weight at (row i, col j) within block
            const std::int8_t* block = &B_bsr.data[idx * BSR_BLOCK_ELEMENTS];
            
            // For each row in the activation matrix A
            // Hardware: This is the "M-tile" loop in the scheduler
            for (std::size_t m = 0; m < M; ++m) {
                
                // For each column within the block (0..13)
                // Hardware: These 14 outputs computed in parallel by 14 PE columns
                for (std::size_t j = 0; j < BSR_BLOCK_SIZE; ++j) {
                    
                    // Map block-local column j to global column index n
                    // n = bc * 14 + j (block_column × block_size + offset)
                    std::size_t n = bc * BSR_BLOCK_SIZE + j;
                    
                    // Bounds check: skip if block extends past N (edge case)
                    // Hardware: mask_valid signal handles partial blocks
                    if (n >= N) continue;
                    
                    // Accumulate the partial sum for C[m][n]
                    // This is the inner product of A[m][br*14 : br*14+14] with block[:][j]
                    std::int32_t acc = 0;
                    
                    // For each row within the block (0..13) - the K reduction
                    // Hardware: 14 cycles to process one block (pipelined)
                    for (std::size_t i = 0; i < BSR_BLOCK_SIZE; ++i) {
                        
                        // Map block-local row i to global K index
                        // k = br * 14 + i (block_row × block_size + offset)
                        std::size_t k = br * BSR_BLOCK_SIZE + i;
                        
                        // Bounds check for partial blocks at edge of K dimension
                        if (k >= K) continue;
                        
                        // The core MAC operation (identical to dense matmul)
                        // Hardware: mac8.sv computes int8 × int8 + int32 → int32
                        acc += static_cast<std::int32_t>(A[m * K + k]) *
                               static_cast<std::int32_t>(block[i * BSR_BLOCK_SIZE + j]);
                    }
                    
                    // Accumulate into output (+=, not =, because multiple blocks
                    // may contribute to the same output element across block-rows)
                    // Hardware: output_accumulator.sv handles this accumulation
                    C[m * N + n] += acc;
                }
            }
        }
    }
}

//==============================================================================
// ACTIVATION FUNCTIONS
//==============================================================================
/**
 * ReLU (Rectified Linear Unit) - INT8 In-Place
 *
 * Formula: output[i] = max(0, input[i])
 *
 * Hardware Implementation:
 *   - Typically fused with requantization in output_accumulator.sv
 *   - Single-cycle comparison and mux: (val < 0) ? 0 : val
 *   - No additional latency when fused
 *
 * Why In-Place:
 *   - Saves memory allocation
 *   - Better cache utilization (data stays hot)
 *   - Matches hardware behavior (fused with accumulator readout)
 *
 * @param data  INT8 buffer to apply ReLU to (modified in-place)
 * @param size  Number of elements
 */
void relu_int8(std::int8_t* data, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        // INT8 comparison: -128 to 127, negative values become 0
        // Hardware: single LUT implementing (data[7] == 1) ? 0 : data
        if (data[i] < 0) data[i] = 0;
    }
}

/**
 * ReLU on INT32 Data (pre-requantization)
 *
 * Used when ReLU is applied to accumulator values before requantization.
 * This is more accurate than applying ReLU after quantization because
 * we haven't lost precision yet.
 *
 * Hardware: Applied in output_accumulator.sv before scale/shift
 *
 * @param data  INT32 accumulator values (modified in-place)
 * @param size  Number of elements
 */
void relu_int32(std::int32_t* data, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        // INT32 comparison for accumulator values
        if (data[i] < 0) data[i] = 0;
    }
}

/**
 * ReLU6 - Clamped ReLU (used in MobileNet, EfficientNet)
 *
 * Formula: output[i] = min(max(0, input[i]), 6/scale)
 *
 * The "6" in ReLU6 refers to the floating-point value 6.0.
 * In quantized form: max_val = 6.0 / scale
 *
 * Example:
 *   scale = 0.05 → max_val = 6.0 / 0.05 = 120
 *   So output is clamped to [0, 120] in INT8
 *
 * Hardware: Additional comparator in output stage
 *
 * @param data   INT8 buffer to apply ReLU6 to (modified in-place)
 * @param size   Number of elements
 * @param scale  Quantization scale (float_val = int_val × scale)
 */
void relu6_int8(std::int8_t* data, std::size_t size, float scale) {
    // Convert floating-point 6.0 to quantized INT8 threshold
    // max_val = 6.0 / scale (truncate, don't round, to be conservative)
    std::int8_t max_val = static_cast<std::int8_t>(6.0f / scale);
    
    for (std::size_t i = 0; i < size; ++i) {
        // Lower bound: ReLU (clamp negatives to 0)
        if (data[i] < 0) data[i] = 0;
        // Upper bound: clamp to 6.0 equivalent
        if (data[i] > max_val) data[i] = max_val;
    }
}

//==============================================================================
// REQUANTIZATION (INT32 → INT8)
//==============================================================================
/**
 * Requantize INT32 Accumulator to INT8 for Next Layer
 *
 * After GEMM, we have INT32 accumulator values. Before the next layer,
 * we need to convert back to INT8. This is requantization.
 *
 * Mathematical Derivation:
 *   input (INT32) represents: float_val = input × in_scale
 *   output (INT8) should represent: float_val = output × out_scale
 *   
 *   Therefore: output = input × in_scale / out_scale
 *                     = input × (in_scale / out_scale)
 *                     = input × scale_factor
 *
 * Where in_scale = act_scale × wgt_scale (product of input quantization scales)
 *
 * Rounding Mode: Round-Half-to-Even (Banker's Rounding)
 *   - 0.5 rounds to 0, 1.5 rounds to 2, 2.5 rounds to 2, 3.5 rounds to 4
 *   - Reduces systematic bias compared to round-half-up
 *   - Hardware implements this with a simple rounding circuit
 *   - C++ uses std::nearbyint() with FE_TONEAREST mode
 *
 * Saturation (Clamping):
 *   - INT8 range: [-128, +127]
 *   - Values outside this range are clamped (not wrapped!)
 *   - Wrapping would cause catastrophic errors in the network
 *
 * Hardware Implementation (output_accumulator.sv):
 *   1. Multiply INT32 by fixed-point scale (shift + multiply)
 *   2. Add rounding constant (0.5 in fixed-point)
 *   3. Shift right to get result
 *   4. Saturate to INT8 range
 *
 * @param input     INT32 accumulator values
 * @param output    INT8 quantized output (caller allocates)
 * @param size      Number of elements to process
 * @param in_scale  Product of activation and weight scales
 * @param out_scale Target scale for output layer
 */
void requantize_int32_to_int8(const std::int32_t* input, std::int8_t* output,
                               std::size_t size, float in_scale, float out_scale) {
    // Set IEEE 754 rounding mode to round-half-to-even (banker's rounding)
    // This MUST match hardware behavior for bit-exact results
    // Hardware equivalent: round_bit = frac[MSB] && (frac[MSB-1:0] != 0 || int[0])
    std::fesetround(FE_TONEAREST);
    
    // Pre-compute scale factor to avoid division in loop
    // scale_factor = in_scale / out_scale
    // Example: in_scale = 0.01 (act) × 0.02 (wgt) = 0.0002
    //          out_scale = 0.03 (next layer)
    //          scale_factor = 0.0002 / 0.03 = 0.00667
    float scale_factor = in_scale / out_scale;
    
    for (std::size_t i = 0; i < size; ++i) {
        // Step 1: Scale the INT32 value to the new quantization domain
        // This is a floating-point multiply (hardware uses fixed-point approximation)
        float scaled = static_cast<float>(input[i]) * scale_factor;
        
        // Step 2: Round to nearest integer using banker's rounding
        // std::nearbyint respects the current rounding mode (FE_TONEAREST)
        // Hardware: implemented as add 0.5 + truncate with tie-breaking logic
        std::int32_t rounded = static_cast<std::int32_t>(std::nearbyint(scaled));

        // Step 3: Saturate to INT8 range [-128, +127]
        // CRITICAL: Use saturation (clamping), NOT wrapping (modulo)!
        // Wrapping would turn +128 into -128, completely wrong for the network
        if (rounded > 127) rounded = 127;    // Positive saturation
        if (rounded < -128) rounded = -128;  // Negative saturation
        
        // Step 4: Store as INT8
        output[i] = static_cast<std::int8_t>(rounded);
    }
}

//==============================================================================
// RESIDUAL ADDITION (Skip Connections)
//==============================================================================
/**
 * Add Residual Connection for ResNet Skip Paths
 *
 * ResNet uses skip connections that add the input of a block directly to
 * its output. The challenge: input and output may have DIFFERENT quantization
 * scales, so we can't just add the INT8 values directly.
 *
 * ResNet Block Structure:
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │  input (INT8, scale=0.05)                                       │
 *   │    │                                                            │
 *   │    ├──────────────────────────────────┐ (skip connection)       │
 *   │    │                                  │                         │
 *   │    ▼                                  │                         │
 *   │  [Conv1] → [BN] → [ReLU]              │                         │
 *   │    │                                  │                         │
 *   │    ▼                                  │                         │
 *   │  [Conv2] → [BN]                       │                         │
 *   │    │                                  │                         │
 *   │    ▼ (INT8, scale=0.03)               │                         │
 *   │    ├───────────────────────────────── + ← residual addition     │
 *   │    │                                                            │
 *   │    ▼                                                            │
 *   │  [ReLU]                                                         │
 *   │    │                                                            │
 *   │    ▼                                                            │
 *   │  output (INT8, scale=0.03)                                      │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * Addition Algorithm:
 *   1. Dequantize both paths to float: float_val = int_val × scale
 *   2. Add in floating-point domain (no precision loss)
 *   3. Requantize result to output scale
 *
 * Hardware Implementation Options:
 *   A. Floating-point addition (expensive, high latency)
 *   B. Fixed-point rescaling: rescale one operand before integer add
 *   C. Lookup table for common scale ratios
 *
 * Current implementation uses approach (A) for simplicity/accuracy.
 *
 * @param main           Main path output [size], INT8
 * @param residual       Skip connection input [size], INT8
 * @param output         Combined output [size], INT8
 * @param size           Number of elements
 * @param main_scale     Quantization scale for main path
 * @param residual_scale Quantization scale for skip connection
 * @param out_scale      Target scale for output
 */
void add_residual_int8(const std::int8_t* main, const std::int8_t* residual,
                       std::int8_t* output, std::size_t size,
                       float main_scale, float residual_scale, float out_scale) {
    for (std::size_t i = 0; i < size; ++i) {
        // Step 1: Dequantize both inputs to floating-point
        // This "undoes" the quantization: float = int × scale
        float main_val = static_cast<float>(main[i]) * main_scale;
        float res_val = static_cast<float>(residual[i]) * residual_scale;
        
        // Step 2: Add in floating-point domain
        // Now both values are in the same "units" (actual floating-point)
        float sum = main_val + res_val;
        
        // Step 3: Requantize to output scale
        // This converts back to INT8: int = round(float / scale)
        std::int32_t quantized = static_cast<std::int32_t>(
            std::nearbyint(sum / out_scale));
        
        // Step 4: Saturate to INT8 range
        // Same saturation logic as requantization
        if (quantized > 127) quantized = 127;
        if (quantized < -128) quantized = -128;
        
        output[i] = static_cast<std::int8_t>(quantized);
    }
}

//==============================================================================
// POOLING OPERATIONS
//==============================================================================
/**
 * 2D Max Pooling - INT8
 *
 * Max pooling reduces spatial dimensions by taking the maximum value
 * in each pooling window. Common configuration: 2×2 pool with stride 2,
 * which halves both H and W.
 *
 * Pooling Window Example (2×2):
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │  Input (4×4):          After 2×2 maxpool:                       │
 *   │  ┌───┬───┬───┬───┐     ┌───┬───┐                               │
 *   │  │ 1 │ 3 │ 5 │ 2 │     │ 4 │ 6 │  max(1,3,2,4)=4, max(5,2,1,6)=6│
 *   │  ├───┼───┼───┼───┤     ├───┼───┤                               │
 *   │  │ 2 │ 4 │ 1 │ 6 │ ──► │ 8 │ 9 │  max(7,8,3,1)=8, max(4,9,2,5)=9│
 *   │  ├───┼───┼───┼───┤     └───┴───┘                               │
 *   │  │ 7 │ 8 │ 4 │ 9 │                                              │
 *   │  ├───┼───┼───┼───┤                                              │
 *   │  │ 3 │ 1 │ 2 │ 5 │                                              │
 *   │  └───┴───┴───┴───┘                                              │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * Memory Layout: CHW (channel-first, height, width)
 *   input[c][h][w] = input[c * H * W + h * W + w]
 *
 * Output Dimensions:
 *   H_out = (H - pool_size) / stride + 1
 *   W_out = (W - pool_size) / stride + 1
 *
 * Quantization Note:
 *   Max pooling is quantization-preserving: the output is just one of
 *   the input values, so no requantization is needed. Scale is unchanged.
 *
 * @param input      Input tensor [C × H × W], INT8, CHW format
 * @param output     Output tensor [C × H_out × W_out], INT8
 * @param H, W       Input spatial dimensions
 * @param C          Number of channels
 * @param pool_size  Pooling window size (typically 2)
 * @param stride     Stride between windows (typically 2)
 */
void maxpool2d_int8(const std::int8_t* input, std::int8_t* output,
                    std::size_t H, std::size_t W, std::size_t C,
                    std::size_t pool_size, std::size_t stride) {
    // Calculate output dimensions
    std::size_t H_out = (H - pool_size) / stride + 1;
    std::size_t W_out = (W - pool_size) / stride + 1;
    
    // Iterate over channels (pooling is per-channel, independent)
    for (std::size_t c = 0; c < C; ++c) {
        // Iterate over output positions
        for (std::size_t oh = 0; oh < H_out; ++oh) {
            for (std::size_t ow = 0; ow < W_out; ++ow) {
                // Initialize max to minimum INT8 value
                // Hardware: register initialized to 0x80 (-128)
                std::int8_t max_val = -128;
                
                // Scan the pooling window
                for (std::size_t ph = 0; ph < pool_size; ++ph) {
                    for (std::size_t pw = 0; pw < pool_size; ++pw) {
                        // Map output position to input position
                        std::size_t h = oh * stride + ph;
                        std::size_t w = ow * stride + pw;
                        
                        // Get input value (CHW layout)
                        std::int8_t val = input[c * H * W + h * W + w];
                        
                        // Update maximum
                        // Hardware: simple comparator + mux
                        if (val > max_val) max_val = val;
                    }
                }
                
                // Store the maximum value for this window
                output[c * H_out * W_out + oh * W_out + ow] = max_val;
            }
        }
    }
}

/**
 * Global Average Pooling - INT8
 *
 * Reduces each channel to a single value by averaging all spatial elements.
 * Used in ResNet before the final FC layer to reduce [C × H × W] to [C].
 *
 * Example:
 *   Input: [512 × 7 × 7] (after final conv block)
 *   Output: [512] (one value per channel)
 *
 * Rounding Strategy:
 *   Average with rounding: avg = (sum + H*W/2) / (H*W)
 *   The "+H*W/2" provides round-half-up behavior.
 *   
 *   Example: sum=100, H*W=49
 *     Without rounding: 100/49 = 2.04 → 2
 *     With rounding: (100+24)/49 = 124/49 = 2.53 → 2 (integer division)
 *     Actually: round(100/49) = round(2.04) = 2 ✓
 *
 * Quantization Note:
 *   The output scale remains unchanged because we're just computing
 *   an average of values with the same scale.
 *
 * @param input   Input tensor [C × H × W], INT8, CHW format
 * @param output  Output tensor [C], INT8
 * @param H, W    Input spatial dimensions
 * @param C       Number of channels
 */
void avgpool_global_int8(const std::int8_t* input, std::int8_t* output,
                         std::size_t H, std::size_t W, std::size_t C) {
    for (std::size_t c = 0; c < C; ++c) {
        // Sum all spatial elements for this channel
        // Use INT32 to avoid overflow (max: 127 × H × W)
        // For H=W=224: 127 × 224 × 224 = 6,373,376 (fits in INT32)
        std::int32_t sum = 0;
        
        for (std::size_t h = 0; h < H; ++h) {
            for (std::size_t w = 0; w < W; ++w) {
                sum += static_cast<std::int32_t>(input[c * H * W + h * W + w]);
            }
        }
        
        // Compute average with rounding
        // Formula: avg = (sum + divisor/2) / divisor
        // This implements round-half-up (0.5 rounds to 1)
        // Hardware: add half, then divide (or shift if power of 2)
        std::int32_t avg = (sum + static_cast<std::int32_t>(H * W / 2)) / 
                           static_cast<std::int32_t>(H * W);
        
        // Saturate to INT8 range (unlikely to be needed for average)
        if (avg > 127) avg = 127;
        if (avg < -128) avg = -128;
        
        output[c] = static_cast<std::int8_t>(avg);
    }
}

//==============================================================================
// CONVOLUTION - SIMPLE DIRECT METHOD (6 Nested Loops)
//==============================================================================
/**
 * 2D Convolution with INT8 Arithmetic - Direct Method
 *
 * This is the "textbook" implementation with 6 nested loops.
 * It's easy to understand but SLOW due to poor memory access patterns.
 * Used only for reference/verification, not for performance.
 *
 * Convolution Operation:
 *   For each output position (c_out, oh, ow):
 *     output[c_out][oh][ow] = bias[c_out] +
 *       Σ(c_in, kh, kw) input[c_in][oh*s+kh-p][ow*s+kw-p] × weight[c_out][c_in][kh][kw]
 *
 * Loop Structure (6 levels):
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │ 1. c_out (output channel): Which filter we're applying (0..C_out)   │
 *   │    2. oh (output height): Output row position (0..H_out)            │
 *   │       3. ow (output width): Output column position (0..W_out)       │
 *   │          4. c_in (input channel): Sum across input channels         │
 *   │             5. kh (kernel height): Kernel row (0..K)                │
 *   │                6. kw (kernel width): Kernel column (0..K)           │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 * Memory Layout:
 *   Input:  [C_in × H × W] (CHW format, row-major)
 *   Weight: [C_out × C_in × K × K] (NCHW format)
 *   Output: [C_out × H_out × W_out] (CHW format)
 *
 * Output Dimensions:
 *   H_out = (H + 2×padding - K) / stride + 1
 *   W_out = (W + 2×padding - K) / stride + 1
 *
 * Padding Handling:
 *   Out-of-bounds input positions are treated as 0 (zero-padding).
 *   This is standard for CNNs to preserve spatial dimensions.
 *
 * Why This Is Slow:
 *   - Poor cache utilization (random access patterns in inner loops)
 *   - No vectorization opportunity
 *   - Redundant loads of the same input values
 *
 * @param input    Input tensor [C_in × H × W], INT8
 * @param weight   Weight tensor [C_out × C_in × K × K], INT8
 * @param bias     Bias vector [C_out], INT32 (or nullptr if no bias)
 * @param output   Output tensor [C_out × H_out × W_out], INT32
 * @param C_in     Number of input channels
 * @param H, W     Input spatial dimensions
 * @param C_out    Number of output channels (number of filters)
 * @param K        Kernel size (K×K, typically 3 for ResNet)
 * @param stride   Convolution stride (typically 1 or 2)
 * @param padding  Zero-padding amount (typically K/2 for 'same' padding)
 */
void conv2d_int8_simple(const std::int8_t* input, const std::int8_t* weight,
                        const std::int32_t* bias, std::int32_t* output,
                        std::size_t C_in, std::size_t H, std::size_t W,
                        std::size_t C_out, std::size_t K,
                        std::size_t stride, std::size_t padding) {
    
    // Calculate output spatial dimensions
    // Formula: O = (I + 2P - K) / S + 1
    // Example: H=224, K=3, stride=1, padding=1 → H_out = (224+2-3)/1+1 = 224
    std::size_t H_out = (H + 2 * padding - K) / stride + 1;
    std::size_t W_out = (W + 2 * padding - K) / stride + 1;
    
    // Loop 1: For each output channel (each filter)
    // Each filter produces one channel of output
    for (std::size_t c_out = 0; c_out < C_out; ++c_out) {
        
        // Loop 2 & 3: For each output spatial position
        for (std::size_t oh = 0; oh < H_out; ++oh) {
            for (std::size_t ow = 0; ow < W_out; ++ow) {
                
                // Start with bias (or 0 if no bias)
                // Bias is INT32 to match accumulator precision
                std::int32_t acc = bias ? bias[c_out] : 0;
                
                // Loop 4: Sum contributions from all input channels
                for (std::size_t c_in = 0; c_in < C_in; ++c_in) {
                    
                    // Loop 5 & 6: Slide kernel over input
                    for (std::size_t kh = 0; kh < K; ++kh) {
                        for (std::size_t kw = 0; kw < K; ++kw) {
                            
                            // Calculate input position with stride and padding
                            // ih = oh × stride + kh - padding
                            // This maps output position to input position
                            std::int64_t ih = oh * stride + kh - padding;
                            std::int64_t iw = ow * stride + kw - padding;
                            
                            // Check for zero-padding (out of bounds = 0)
                            // Negative or >= dimension means we're in the padding region
                            if (ih < 0 || ih >= static_cast<std::int64_t>(H) || 
                                iw < 0 || iw >= static_cast<std::int64_t>(W)) {
                                continue;  // Zero-padding: contribute 0 to sum
                            }
                            
                            // Get input value at (c_in, ih, iw)
                            // CHW layout: input[c_in * H * W + ih * W + iw]
                            std::int8_t in_val = input[c_in * H * W + ih * W + iw];
                            
                            // Get weight value at (c_out, c_in, kh, kw)
                            // OIHW layout: weight[((c_out * C_in + c_in) * K + kh) * K + kw]
                            std::int8_t wt_val = weight[((c_out * C_in + c_in) * K + kh) * K + kw];
                            
                            // Multiply and accumulate (MAC)
                            // Cast to INT32 before multiply to prevent overflow
                            acc += static_cast<std::int32_t>(in_val) * 
                                   static_cast<std::int32_t>(wt_val);
                        }
                    }
                }
                
                // Store accumulated result
                // Output layout: [C_out × H_out × W_out]
                output[c_out * H_out * W_out + oh * W_out + ow] = acc;
            }
        }
    }
}

//==============================================================================
// CONVOLUTION - IM2COL + GEMM METHOD (Optimized)
//==============================================================================
/**
 * im2col: Image-to-Column Transformation
 *
 * This is the key optimization that makes convolution efficient on hardware.
 * Instead of doing convolution directly, we:
 *   1. Reshape input patches into columns (im2col)
 *   2. Reshape weights into a 2D matrix
 *   3. Perform matrix multiplication (GEMM)
 *   4. Reshape result back to output tensor
 *
 * Why im2col is faster:
 *   - Matrix multiplication is highly optimized (SIMD, cache-friendly)
 *   - Can reuse our systolic array GEMM hardware!
 *   - Avoids redundant memory accesses
 *   - Industry standard: used by cuDNN, MKL-DNN, etc.
 *
 * Transformation Visualization:
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │  Input [C_in × H × W]           im2col buffer [patch_size × patches]│
 *   │  ┌─────────────┐                ┌─────────────────────┐             │
 *   │  │ Channel 0   │                │ p0  p1  p2 ... pN-1 │             │
 *   │  │ ┌───────┐   │                │  │   │   │       │  │             │
 *   │  │ │ patch │   │  ──im2col──►   │  ▼   ▼   ▼       ▼  │             │
 *   │  │ │  0    │   │                │ [flattened patches] │             │
 *   │  │ └───────┘   │                │ [C_in × K × K rows] │             │
 *   │  └─────────────┘                │ [H_out×W_out cols]  │             │
 *   │  ...                            └─────────────────────┘             │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 *   Then: Output = Weight × im2col_buffer
 *         [C_out × patches] = [C_out × patch_size] × [patch_size × patches]
 *
 * Memory Layout of im2col buffer:
 *   - Rows: C_in × K × K (flattened patch: all channels, all kernel positions)
 *   - Columns: H_out × W_out (one column per output position)
 *   - Each column is one complete input patch
 *
 * @param input       Input tensor [C_in × H × W], INT8
 * @param col_buffer  Output buffer [C_in×K×K × H_out×W_out], INT8
 * @param C_in        Number of input channels
 * @param H, W        Input spatial dimensions
 * @param K           Kernel size
 * @param stride      Convolution stride
 * @param padding     Zero-padding amount
 * @param H_out, W_out Output spatial dimensions (must be pre-computed)
 */
void im2col_int8(const std::int8_t* input,
                 std::int8_t* col_buffer,
                 std::size_t C_in, std::size_t H, std::size_t W,
                 std::size_t K, std::size_t stride, std::size_t padding,
                 std::size_t H_out, std::size_t W_out) {
    // col_buffer shape: [C_in × K × K] rows × [H_out × W_out] columns
    // Each column represents one flattened input patch
    
    std::size_t col_idx = 0;  // Column index (which output position)
    
    // Iterate over output positions (each becomes one column)
    for (std::size_t oh = 0; oh < H_out; ++oh) {
        for (std::size_t ow = 0; ow < W_out; ++ow) {
            std::size_t row_idx = 0;  // Row index within the column
            
            // For each input channel (contributes K×K elements to patch)
            for (std::size_t c_in = 0; c_in < C_in; ++c_in) {
                // For each kernel position (K×K)
                for (std::size_t kh = 0; kh < K; ++kh) {
                    for (std::size_t kw = 0; kw < K; ++kw) {
                        // Calculate corresponding input position
                        std::int64_t h = oh * stride + kh - padding;
                        std::int64_t w = ow * stride + kw - padding;
                        
                        // Get value (0 if out of bounds = zero-padding)
                        std::int8_t val = 0;
                        if (h >= 0 && h < static_cast<std::int64_t>(H) && 
                            w >= 0 && w < static_cast<std::int64_t>(W)) {
                            val = input[c_in * H * W + h * W + w];
                        }
                        
                        // Store in column-major layout for efficient GEMM
                        // col_buffer[row][col] = col_buffer[row × num_cols + col]
                        col_buffer[row_idx * (H_out * W_out) + col_idx] = val;
                        row_idx++;
                    }
                }
            }
            col_idx++;
        }
    }
}

/**
 * Full im2col Convolution: Transform + GEMM
 *
 * This is how convolution is actually implemented efficiently.
 * The key insight is that convolution can be expressed as matrix multiplication:
 *
 *   output = weight × im2col(input)
 *
 * Where:
 *   - weight is reshaped to [C_out × (C_in × K × K)]
 *   - im2col(input) is [(C_in × K × K) × (H_out × W_out)]
 *   - output is [C_out × (H_out × W_out)]
 *
 * Hardware Mapping:
 *   The im2col buffer is what we load into act_buffer.
 *   The reshaped weights go into wgt_buffer.
 *   The systolic array performs the GEMM.
 *   Output is reshaped back to [C_out × H_out × W_out].
 *
 * Performance:
 *   This is ~10× faster than the direct method for typical CNN layers
 *   due to better cache utilization and vectorization opportunities.
 *
 * Memory Overhead:
 *   im2col buffer size = C_in × K × K × H_out × W_out bytes
 *   For ResNet-18 conv1: 3 × 7 × 7 × 112 × 112 = 1.8 MB
 *   Tradeoff: More memory for much better performance
 *
 * @param input    Input tensor [C_in × H × W], INT8
 * @param weight   Weight tensor [C_out × C_in × K × K], INT8 (row-major)
 * @param bias     Bias vector [C_out], INT32 (or nullptr)
 * @param output   Output tensor [C_out × H_out × W_out], INT32
 * @param C_in     Number of input channels
 * @param H, W     Input spatial dimensions
 * @param C_out    Number of output channels
 * @param K        Kernel size
 * @param stride   Convolution stride
 * @param padding  Zero-padding amount
 */
void conv2d_int8_im2col(const std::int8_t* input, const std::int8_t* weight,
                        const std::int32_t* bias, std::int32_t* output,
                        std::size_t C_in, std::size_t H, std::size_t W,
                        std::size_t C_out, std::size_t K,
                        std::size_t stride, std::size_t padding) {
    // Calculate output dimensions
    std::size_t H_out = (H + 2 * padding - K) / stride + 1;
    std::size_t W_out = (W + 2 * padding - K) / stride + 1;
    
    // Allocate im2col buffer
    // Shape: [C_in × K × K] rows × [H_out × W_out] columns
    std::size_t col_rows = C_in * K * K;      // Flattened patch size
    std::size_t col_cols = H_out * W_out;     // Number of output positions
    std::vector<std::int8_t> col_buffer(col_rows * col_cols);
    
    // Step 1: Transform input using im2col
    // This rearranges input patches into columns for efficient GEMM
    im2col_int8(input, col_buffer.data(), C_in, H, W, K, stride, padding, H_out, W_out);
    
    // Step 2: Matrix multiplication (GEMM)
    // Weight shape:     [C_out] × [C_in × K × K] (already in this format)
    // Col buffer shape: [C_in × K × K] × [H_out × W_out]
    // Output shape:     [C_out] × [H_out × W_out]
    //
    // This is: output = weight × col_buffer
    // Which computes all output positions in one GEMM!
    
    for (std::size_t c_out = 0; c_out < C_out; ++c_out) {
        for (std::size_t col = 0; col < col_cols; ++col) {
            // Start with bias for this output channel
            std::int32_t acc = bias ? bias[c_out] : 0;
            
            // Dot product: one row of weight × one column of col_buffer
            // This is equivalent to the 4 inner loops of the direct method!
            for (std::size_t k = 0; k < col_rows; ++k) {
                // Weight is [C_out × col_rows], row-major
                std::int8_t wt_val = weight[c_out * col_rows + k];
                // Col buffer is [col_rows × col_cols], row-major
                std::int8_t in_val = col_buffer[k * col_cols + col];
                
                // MAC operation
                acc += static_cast<std::int32_t>(wt_val) * 
                       static_cast<std::int32_t>(in_val);
            }
            
            // Store result
            // Output is [C_out × col_cols] = [C_out × H_out × W_out]
            output[c_out * col_cols + col] = acc;
        }
    }
}

} // namespace golden
} // namespace resnet_accel