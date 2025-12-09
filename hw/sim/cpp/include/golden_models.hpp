/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                        GOLDEN_MODELS.HPP                                  ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  REPLACES: sw/golden_models/conv_golden.py                               ║
 * ║            sw/golden_models/matmul_golden.py                             ║
 * ║            sw/golden/sparse_matmul.py                                    ║
 * ║            Any other Python golden/reference implementations             ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║    Bit-exact reference implementations of all accelerator operations.    ║
 * ║    These produce the EXACT same results as hardware, bit-for-bit.        ║
 * ║    Used to verify RTL correctness in simulation.                         ║
 * ║                                                                           ║
 * ║  WHY C++ INSTEAD OF PYTHON:                                               ║
 * ║    • Python floats introduce numerical differences                       ║
 * ║    • C++ int8_t/int32_t arithmetic matches hardware exactly              ║
 * ║    • Same accumulation order as systolic array dataflow                  ║
 * ║    • Can run millions of test vectors quickly                            ║
 * ║    • Integrates directly with Verilator for co-simulation                ║
 * ║                                                                           ║
 * ║  CRITICAL: ACCUMULATION ORDER MATTERS!                                   ║
 * ║    The systolic array accumulates in a specific order due to its         ║
 * ║    pipelined dataflow. Your golden model MUST match this order           ║
 * ║    exactly, or you'll get "correct" results that don't match hardware.   ║
 * ║                                                                           ║
 * ║    Row-stationary dataflow accumulation:                                 ║
 * ║    ┌─────────────────────────────────────────────────────────────────┐   ║
 * ║    │  For output[m][n]:                                              │   ║
 * ║    │    acc = 0                                                      │   ║
 * ║    │    for k in range(K):        # K dimension (reduction)          │   ║
 * ║    │      acc += A[m][k] * B[k][n]  # INT8 * INT8 -> INT32          │   ║
 * ║    │    output[m][n] = acc                                           │   ║
 * ║    └─────────────────────────────────────────────────────────────────┘   ║
 * ║                                                                           ║
 * ║  FUNCTIONS TO IMPLEMENT:                                                  ║
 * ║                                                                           ║
 * ║  ┌─────────────────────────────────────────────────────────────────────┐ ║
 * ║  │ MATRIX OPERATIONS                                                   │ ║
 * ║  ├─────────────────────────────────────────────────────────────────────┤ ║
 * ║  │ golden_matmul_int8(A, B, C, M, K, N)                               │ ║
 * ║  │   - Dense matrix multiply: C[M,N] = A[M,K] @ B[K,N]                │ ║
 * ║  │   - INT8 inputs, INT32 output (accumulator width)                  │ ║
 * ║  │                                                                     │ ║
 * ║  │ golden_bsr_matmul_int8(A, B_bsr, C, M, K, N)                       │ ║
 * ║  │   - Sparse matrix multiply using BSR format for B                  │ ║
 * ║  │   - Only computes non-zero blocks                                  │ ║
 * ║  │   - THIS IS THE KEY FUNCTION - must match hardware exactly         │ ║
 * ║  └─────────────────────────────────────────────────────────────────────┘ ║
 * ║                                                                           ║
 * ║  ┌─────────────────────────────────────────────────────────────────────┐ ║
 * ║  │ CONVOLUTION OPERATIONS                                              │ ║
 * ║  ├─────────────────────────────────────────────────────────────────────┤ ║
 * ║  │ golden_conv2d_int8(input, weight, bias, output, ...)               │ ║
 * ║  │   - Standard 2D convolution with INT8 arithmetic                   │ ║
 * ║  │   - Parameters: batch, in/out channels, H, W, kernel, stride, pad  │ ║
 * ║  │   - Uses im2col + matmul internally (like hardware)                │ ║
 * ║  │                                                                     │ ║
 * ║  │ golden_conv2d_bsr_int8(input, weight_bsr, bias, output, ...)       │ ║
 * ║  │   - Sparse convolution using BSR weight format                     │ ║
 * ║  └─────────────────────────────────────────────────────────────────────┘ ║
 * ║                                                                           ║
 * ║  ┌─────────────────────────────────────────────────────────────────────┐ ║
 * ║  │ ACTIVATION FUNCTIONS                                                │ ║
 * ║  ├─────────────────────────────────────────────────────────────────────┤ ║
 * ║  │ golden_relu_int8(data, size)                                       │ ║
 * ║  │   - In-place ReLU: data[i] = max(0, data[i])                       │ ║
 * ║  │                                                                     │ ║
 * ║  │ golden_relu6_int8(data, size, scale)                               │ ║
 * ║  │   - Clamped ReLU: data[i] = clamp(data[i], 0, 6/scale)             │ ║
 * ║  └─────────────────────────────────────────────────────────────────────┘ ║
 * ║                                                                           ║
 * ║  ┌─────────────────────────────────────────────────────────────────────┐ ║
 * ║  │ POOLING OPERATIONS                                                  │ ║
 * ║  ├─────────────────────────────────────────────────────────────────────┤ ║
 * ║  │ golden_maxpool2d_int8(input, output, H, W, C, pool_size, stride)   │ ║
 * ║  │   - 2D max pooling (typically 2x2 with stride 2)                   │ ║
 * ║  │                                                                     │ ║
 * ║  │ golden_avgpool_global_int8(input, output, H, W, C)                 │ ║
 * ║  │   - Global average pooling (used before FC layer in ResNet)        │ ║
 * ║  └─────────────────────────────────────────────────────────────────────┘ ║
 * ║                                                                           ║
 * ║  ┌─────────────────────────────────────────────────────────────────────┐ ║
 * ║  │ RESIDUAL OPERATIONS                                                 │ ║
 * ║  ├─────────────────────────────────────────────────────────────────────┤ ║
 * ║  │ golden_add_residual_int8(main, residual, output, size, ...)        │ ║
 * ║  │   - Element-wise addition for skip connections                     │ ║
 * ║  │   - Must handle different quantization scales!                     │ ║
 * ║  │   - Formula: out = requant(dequant(main) + dequant(residual))      │ ║
 * ║  └─────────────────────────────────────────────────────────────────────┘ ║
 * ║                                                                           ║
 * ║  ┌─────────────────────────────────────────────────────────────────────┐ ║
 * ║  │ QUANTIZATION                                                        │ ║
 * ║  ├─────────────────────────────────────────────────────────────────────┤ ║
 * ║  │ golden_requantize_int32_to_int8(input, output, in_scale, out_scale)│ ║
 * ║  │   - Convert INT32 accumulator to INT8 for next layer               │ ║
 * ║  │   - out = round(in * in_scale / out_scale)                         │ ║
 * ║  │   - Clamp to [-128, 127]                                           │ ║
 * ║  └─────────────────────────────────────────────────────────────────────┘ ║
 * ║                                                                           ║
 * ║  IMPLEMENTATION NOTES:                                                    ║
 * ║    • Use int32_t for accumulation, never int8_t (overflow!)             ║
 * ║    • Watch for signed vs unsigned - weights are signed INT8            ║
 * ║    • Rounding: use round-half-to-even (banker's rounding) like HW      ║
 * ║    • Saturation: clamp don't wrap on overflow                          ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#ifndef GOLDEN_MODELS_HPP
#define GOLDEN_MODELS_HPP

#include <cstdint>
#include <cstddef>
#include "bsr_packer.hpp"  // For BSRMatrix

/**
 * Golden model function declarations
 * 
 * All functions produce bit-exact results matching hardware
 */
namespace golden {

// =============================================================================
// Matrix Operations
// =============================================================================

/**
 * Dense matrix multiply: C = A @ B
 * 
 * @param A  Input matrix [M x K], row-major, INT8
 * @param B  Input matrix [K x N], row-major, INT8
 * @param C  Output matrix [M x N], row-major, INT32
 * 
 * TODO: Implement with exact accumulation order matching systolic array
 */
void matmul_int8(const int8_t* A, const int8_t* B, int32_t* C,
                 size_t M, size_t K, size_t N);

/**
 * Sparse matrix multiply using BSR format: C = A @ B_bsr
 * 
 * @param A      Dense activation matrix [M x K], row-major, INT8
 * @param B_bsr  Sparse weight matrix in BSR format
 * @param C      Output matrix [M x N], row-major, INT32
 * 
 * TODO: Implement - this is the KEY function
 *   For each block row in B_bsr:
 *     For each non-zero block (col_idx, block_data):
 *       Multiply corresponding A columns with block
 *       Accumulate into C
 */
void bsr_matmul_int8(const int8_t* A, const BSRMatrix& B_bsr, int32_t* C,
                     size_t M, size_t K, size_t N);

// =============================================================================
// Convolution Operations
// =============================================================================

/**
 * 2D Convolution with INT8 arithmetic
 * 
 * @param input   [batch, in_channels, in_height, in_width], INT8
 * @param weight  [out_channels, in_channels, kernel, kernel], INT8
 * @param bias    [out_channels], INT32 (or nullptr)
 * @param output  [batch, out_channels, out_height, out_width], INT32
 * 
 * TODO: Implement using im2col + matmul approach:
 *   1. im2col: Reshape input patches into matrix
 *   2. matmul: Multiply with reshaped weights
 *   3. Add bias
 */
void conv2d_int8(const int8_t* input, const int8_t* weight, const int32_t* bias,
                 int32_t* output,
                 size_t batch, size_t in_channels, size_t out_channels,
                 size_t in_height, size_t in_width,
                 size_t kernel_size, size_t stride, size_t padding);

/**
 * 2D Convolution with BSR sparse weights
 */
void conv2d_bsr_int8(const int8_t* input, const BSRMatrix& weight_bsr,
                     const int32_t* bias, int32_t* output,
                     size_t batch, size_t in_channels, size_t out_channels,
                     size_t in_height, size_t in_width,
                     size_t kernel_size, size_t stride, size_t padding);

// =============================================================================
// Activation Functions
// =============================================================================

/**
 * ReLU activation (in-place)
 * data[i] = max(0, data[i])
 */
void relu_int8(int8_t* data, size_t size);

/**
 * ReLU on INT32 data (for use before requantization)
 */
void relu_int32(int32_t* data, size_t size);

/**
 * ReLU6 activation (in-place)
 * data[i] = clamp(data[i], 0, 6/scale)
 */
void relu6_int8(int8_t* data, size_t size, float scale);

// =============================================================================
// Pooling Operations
// =============================================================================

/**
 * 2D Max Pooling
 * 
 * @param input   [C, H, W], INT8
 * @param output  [C, H/stride, W/stride], INT8
 */
void maxpool2d_int8(const int8_t* input, int8_t* output,
                    size_t channels, size_t in_height, size_t in_width,
                    size_t pool_size, size_t stride);

/**
 * Global Average Pooling (for ResNet before FC layer)
 * 
 * @param input   [C, H, W], INT8
 * @param output  [C], INT32 (sum, will be divided later)
 */
void avgpool_global_int8(const int8_t* input, int32_t* output,
                         size_t channels, size_t height, size_t width);

// =============================================================================
// Residual Operations
// =============================================================================

/**
 * Add residual connection
 * 
 * @param main       Main path output [size], INT32
 * @param residual   Skip connection [size], INT8
 * @param output     Combined output [size], INT32
 * @param main_scale      Scale factor for main path
 * @param residual_scale  Scale factor for residual
 * 
 * TODO: Implement scale handling for quantized addition
 */
void add_residual(const int32_t* main, const int8_t* residual, int32_t* output,
                  size_t size, float main_scale, float residual_scale);

// =============================================================================
// Quantization Operations
// =============================================================================

/**
 * Requantize INT32 accumulator to INT8 for next layer
 * 
 * @param input      INT32 accumulator values
 * @param output     INT8 quantized output
 * @param size       Number of elements
 * @param in_scale   Input scale (act_scale * wgt_scale)
 * @param out_scale  Output scale for next layer
 * 
 * Formula: output = round(input * in_scale / out_scale)
 * With clamping to [-128, 127]
 * 
 * TODO: Use round-half-to-even (banker's rounding)
 */
void requantize_int32_to_int8(const int32_t* input, int8_t* output,
                               size_t size, float in_scale, float out_scale);

/**
 * Quantize FP32 to INT8
 */
void quantize_float_to_int8(const float* input, int8_t* output,
                            size_t size, float scale);

/**
 * Dequantize INT8 to FP32
 */
void dequantize_int8_to_float(const int8_t* input, float* output,
                              size_t size, float scale);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Compare two buffers for exact match
 * Returns number of mismatches (0 = perfect match)
 */
size_t compare_buffers_int8(const int8_t* expected, const int8_t* actual,
                            size_t size, bool print_mismatches = false);

size_t compare_buffers_int32(const int32_t* expected, const int32_t* actual,
                             size_t size, bool print_mismatches = false);

/**
 * Compute mean absolute error
 */
float compute_mae_int8(const int8_t* expected, const int8_t* actual, size_t size);

/**
 * im2col transformation (helper for convolution)
 * Transforms input patches into columns for matmul
 */
void im2col_int8(const int8_t* input, int8_t* output,
                 size_t channels, size_t height, size_t width,
                 size_t kernel_size, size_t stride, size_t padding);

} // namespace golden

#endif // GOLDEN_MODELS_HPP
