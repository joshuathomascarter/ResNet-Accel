/**
 * @file golden_models.cpp
 * @brief Bit-exact reference implementations matching hardware
 * @author ResNet-Accel Team
 * @date 2024
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
// Matrix Operations
//==============================================================================

void matmul_int8(const std::int8_t* A, const std::int8_t* B, std::int32_t* C,
                 std::size_t M, std::size_t K, std::size_t N) {
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            std::int32_t acc = 0;
            for (std::size_t k = 0; k < K; ++k) {
                acc += static_cast<std::int32_t>(A[m * K + k]) * 
                       static_cast<std::int32_t>(B[k * N + n]);
            }
            C[m * N + n] = acc;
        }
    }
}

void bsr_matmul_int8(const std::int8_t* A, const BSRMatrix& B_bsr, std::int32_t* C,
                     std::size_t M, std::size_t K, std::size_t N) {
    // Zero initialize output
    std::memset(C, 0, M * N * sizeof(std::int32_t)); //Fully initialises the C full int32 matrix to 0 before computation starts
    
    // Iterate over block rows of B
    for (std::size_t br = 0; br < B_bsr.num_block_rows; ++br) { //goes thru the rows of data in a 4 x 4 itll be 0 -> 1 -> 2 -> 3
        // Iterate over non-zero blocks in this row
        for (std::size_t idx = B_bsr.row_ptr[br]; idx < B_bsr.row_ptr[br + 1]; ++idx) {
            std::size_t bc = B_bsr.col_idx[idx]; // finds the col index in the array of non-zero
            const std::int8_t* block = &B_bsr.data[idx * BSR_BLOCK_ELEMENTS]; //grabs the data for that specific block
            
            // Multiply A columns with this block, accumulate to C
            for (std::size_t m = 0; m < M; ++m) { //iterates thru the A matrix rows 
                for (std::size_t j = 0; j < BSR_BLOCK_SIZE; ++j) { //finds the specific col in the small block (0 to block_size-1)
                    std::size_t n = bc * BSR_BLOCK_SIZE + j; // finds which col we are in the full matrix
                    if (n >= N) continue;

                    
                    std::int32_t acc = 0;
                    for (std::size_t i = 0; i < BSR_BLOCK_SIZE; ++i) {
                        std::size_t k = br * BSR_BLOCK_SIZE + i;
                        if (k >= K) continue;
                        acc += static_cast<std::int32_t>(A[m * K + k]) *
                               static_cast<std::int32_t>(block[i * BSR_BLOCK_SIZE + j]);
                    }
                    C[m * N + n] += acc;
                }
            }
        }
    }
}

//==============================================================================
// Activation Functions
//==============================================================================

void relu_int8(std::int8_t* data, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        if (data[i] < 0) data[i] = 0;
    }
}

void relu_int32(std::int32_t* data, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        if (data[i] < 0) data[i] = 0;
    }
}

void relu6_int8(std::int8_t* data, std::size_t size, float scale) {
    std::int8_t max_val = static_cast<std::int8_t>(6.0f / scale);
    for (std::size_t i = 0; i < size; ++i) {
        if (data[i] < 0) data[i] = 0;
        if (data[i] > max_val) data[i] = max_val;
    }
}

//==============================================================================
// Quantization
//==============================================================================

void requantize_int32_to_int8(const std::int32_t* input, std::int8_t* output,
                               std::size_t size, float in_scale, float out_scale) {
    std::fesetround(FE_TONEAREST);  // Banker's rounding
    float scale_factor = in_scale / out_scale;
    
    for (std::size_t i = 0; i < size; ++i) {
        float scaled = static_cast<float>(input[i]) * scale_factor;
        std::int32_t rounded = static_cast<std::int32_t>(std::nearbyint(scaled));

        // Saturate to INT8 range
        if (rounded > 127) rounded = 127;
        if (rounded < -128) rounded = -128;
        
        output[i] = static_cast<std::int8_t>(rounded);
    }
}

//==============================================================================
// Element-wise Operations
//==============================================================================

void add_residual_int8(const std::int8_t* main, const std::int8_t* residual,
                       std::int8_t* output, std::size_t size,
                       float main_scale, float residual_scale, float out_scale) {
    for (std::size_t i = 0; i < size; ++i) {
        // Dequantize both inputs
        float main_val = static_cast<float>(main[i]) * main_scale;
        float res_val = static_cast<float>(residual[i]) * residual_scale;
        
        // Add in floating point
        float sum = main_val + res_val;
        
        // Requantize to output scale
        std::int32_t quantized = static_cast<std::int32_t>(
            std::nearbyint(sum / out_scale));
        
        // Saturate
        if (quantized > 127) quantized = 127;
        if (quantized < -128) quantized = -128;
        
        output[i] = static_cast<std::int8_t>(quantized);
    }
}

//==============================================================================
// Pooling Operations
//==============================================================================

void maxpool2d_int8(const std::int8_t* input, std::int8_t* output,
                    std::size_t H, std::size_t W, std::size_t C,
                    std::size_t pool_size, std::size_t stride) {
    std::size_t H_out = (H - pool_size) / stride + 1;
    std::size_t W_out = (W - pool_size) / stride + 1;
    
    for (std::size_t c = 0; c < C; ++c) {
        for (std::size_t oh = 0; oh < H_out; ++oh) {
            for (std::size_t ow = 0; ow < W_out; ++ow) {
                std::int8_t max_val = -128;
                
                for (std::size_t ph = 0; ph < pool_size; ++ph) {
                    for (std::size_t pw = 0; pw < pool_size; ++pw) {
                        std::size_t h = oh * stride + ph;
                        std::size_t w = ow * stride + pw;
                        std::int8_t val = input[c * H * W + h * W + w];
                        if (val > max_val) max_val = val;
                    }
                }
                
                output[c * H_out * W_out + oh * W_out + ow] = max_val;
            }
        }
    }
}

void avgpool_global_int8(const std::int8_t* input, std::int8_t* output,
                         std::size_t H, std::size_t W, std::size_t C) {
    for (std::size_t c = 0; c < C; ++c) {
        std::int32_t sum = 0;
        for (std::size_t h = 0; h < H; ++h) {
            for (std::size_t w = 0; w < W; ++w) {
                sum += static_cast<std::int32_t>(input[c * H * W + h * W + w]);
            }
        }
        
        // Average with rounding
        std::int32_t avg = (sum + (H * W / 2)) / (H * W);
        
        // Saturate
        if (avg > 127) avg = 127;
        if (avg < -128) avg = -128;
        
        output[c] = static_cast<std::int8_t>(avg);
    }
}

//==============================================================================
// Convolution (simplified - direct method with 6 nested loops)
//==============================================================================
// 
// This is the "simple" version:
// - Easy to understand
// - Directly computes convolution with nested loops
// - Slow due to poor cache performance
// - Good for reference/verification
//
// Loop structure:
// 1. c_out: output channel (which filter we're applying)
// 2. oh, ow: output position (where in output we're writing)
// 3. c_in: input channel (sum contributions from all input channels)
// 4. kh, kw: kernel position (slide kernel over input)
//
//==============================================================================

void conv2d_int8_simple(const std::int8_t* input, const std::int8_t* weight,
                        const std::int32_t* bias, std::int32_t* output,
                        std::size_t C_in, std::size_t H, std::size_t W,
                        std::size_t C_out, std::size_t K,
                        std::size_t stride, std::size_t padding) {
    
    // Calculate output size
    std::size_t H_out = (H + 2 * padding - K) / stride + 1;
    std::size_t W_out = (W + 2 * padding - K) / stride + 1;
    
    // Loop 1: For each output channel (each filter)
    for (std::size_t c_out = 0; c_out < C_out; ++c_out) {
        
        // Loop 2 & 3: For each output position
        for (std::size_t oh = 0; oh < H_out; ++oh) {
            for (std::size_t ow = 0; ow < W_out; ++ow) {
                
                // Start with bias
                std::int32_t acc = bias ? bias[c_out] : 0;
                
                // Loop 4: For each input channel
                for (std::size_t c_in = 0; c_in < C_in; ++c_in) {
                    
                    // Loop 5 & 6: Slide kernel over input
                    for (std::size_t kh = 0; kh < K; ++kh) {
                        for (std::size_t kw = 0; kw < K; ++kw) {
                            
                            // Find input position
                            std::int64_t ih = oh * stride + kh - padding;
                            std::int64_t iw = ow * stride + kw - padding;
                            
                            // Skip if outside input (padding = 0)
                            if (ih < 0 || ih >= (std::int64_t)H || 
                                iw < 0 || iw >= (std::int64_t)W) {
                                continue;
                            }
                            
                            // Get input and weight values
                            std::int8_t in_val = input[c_in * H * W + ih * W + iw];
                            std::int8_t wt_val = weight[((c_out * C_in + c_in) * K + kh) * K + kw];
                            
                            // Multiply and add
                            acc += (std::int32_t)in_val * (std::int32_t)wt_val;
                        }
                    }
                }
                
                // Store result
                output[c_out * H_out * W_out + oh * W_out + ow] = acc;
            }
        }
    }
}

//==============================================================================
// Convolution (optimized im2col + GEMM version)
//==============================================================================
//
// im2col approach:
// 1. Reorganize input patches into columns (im2col)
// 2. Reshape weights into a 2D matrix
// 3. Perform matrix multiplication (GEMM)
// 4. Result is the convolution output
//
// Why faster?
// - Matrix multiplication is highly optimized (SIMD, cache-friendly)
// - Avoids redundant memory accesses
// - Can use optimized BLAS libraries
//
//==============================================================================

// Helper function: im2col - transform input patches into columns
void im2col_int8(const std::int8_t* input,
                 std::int8_t* col_buffer,
                 std::size_t C_in, std::size_t H, std::size_t W,
                 std::size_t K, std::size_t stride, std::size_t padding,
                 std::size_t H_out, std::size_t W_out) {
    // col_buffer shape: [C_in * K * K] x [H_out * W_out]
    // Each column is one flattened patch of the input
    
    std::size_t col_idx = 0;  // Column index in output
    
    // For each output position
    for (std::size_t oh = 0; oh < H_out; ++oh) {
        for (std::size_t ow = 0; ow < W_out; ++ow) {
            std::size_t row_idx = 0;  // Row index in output
            
            // For each input channel
            for (std::size_t c_in = 0; c_in < C_in; ++c_in) {
                // For each kernel position
                for (std::size_t kh = 0; kh < K; ++kh) {
                    for (std::size_t kw = 0; kw < K; ++kw) {
                        // Calculate input position
                        std::int64_t h = oh * stride + kh - padding;
                        std::int64_t w = ow * stride + kw - padding;
                        
                        // Get value (0 if out of bounds = padding)
                        std::int8_t val = 0;
                        if (h >= 0 && h < static_cast<std::int64_t>(H) && 
                            w >= 0 && w < static_cast<std::int64_t>(W)) {
                            val = input[c_in * H * W + h * W + w];
                        }
                        
                        // Store in col_buffer
                        // Layout: col_buffer[row_idx * (H_out * W_out) + col_idx]
                        col_buffer[row_idx * (H_out * W_out) + col_idx] = val;
                        row_idx++;
                    }
                }
            }
            col_idx++;
        }
    }
}

// Full im2col convolution
void conv2d_int8_im2col(const std::int8_t* input, const std::int8_t* weight,
                        const std::int32_t* bias, std::int32_t* output,
                        std::size_t C_in, std::size_t H, std::size_t W,
                        std::size_t C_out, std::size_t K,
                        std::size_t stride, std::size_t padding) {
    // Calculate output dimensions
    std::size_t H_out = (H + 2 * padding - K) / stride + 1;
    std::size_t W_out = (W + 2 * padding - K) / stride + 1;
    
    // Allocate im2col buffer
    // Shape: [C_in * K * K] x [H_out * W_out]
    std::size_t col_rows = C_in * K * K;      // Flattened patch size
    std::size_t col_cols = H_out * W_out;     // Number of patches
    std::vector<std::int8_t> col_buffer(col_rows * col_cols);
    
    // Step 1: Transform input using im2col
    im2col_int8(input, col_buffer.data(), C_in, H, W, K, stride, padding, H_out, W_out);
    
    // Step 2: Matrix multiplication
    // Weight shape: [C_out] x [C_in * K * K]
    // Col buffer shape: [C_in * K * K] x [H_out * W_out]
    // Output shape: [C_out] x [H_out * W_out]
    //
    // This is: output = weight × col_buffer
    
    for (std::size_t c_out = 0; c_out < C_out; ++c_out) {
        for (std::size_t col = 0; col < col_cols; ++col) {
            // Start with bias
            std::int32_t acc = bias ? bias[c_out] : 0;
            
            // Dot product of weight row and col_buffer column
            for (std::size_t k = 0; k < col_rows; ++k) {
                std::int8_t wt_val = weight[c_out * col_rows + k];
                std::int8_t in_val = col_buffer[k * col_cols + col];
                acc += static_cast<std::int32_t>(wt_val) * static_cast<std::int32_t>(in_val);
            }
            
            // Store result
            output[c_out * col_cols + col] = acc;
        }
    }
}

} // namespace golden
} // namespace resnet_accel