/**
 * @file bsr_packer.hpp
 * @brief Block Sparse Row format packing for sparse matrix acceleration
 * 
 * Converts dense int8 matrices to BSR format with 16×16 blocks optimized
 * for the systolic array hardware. Only non-zero blocks are stored.
 * 
 * Example: 64×64 matrix → 4×4 block grid
 *   row_ptr = [0, 2, 3, 5]      // Cumulative block counts
 *   col_idx = [0, 2, 1, 0, 2]   // Column of each non-zero block
 *   data = [B0_256bytes, B2_256bytes, ...]  // Block values
 * 
 * Usage:
 *   auto bsr = pack_to_bsr(dense, 64, 64);
 *   auto bytes = serialize_for_hardware(bsr);
 *   unpack_from_bsr(bsr, unpacked, 64, 64);
 *   validate_bsr(bsr);
 */

#ifndef BSR_PACKER_HPP
#define BSR_PACKER_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <iostream>

namespace resnet_accel {

//==============================================================================
// Constants
//==============================================================================

/// BSR block size - matches systolic array dimensions (16x16)
constexpr std::size_t BSR_BLOCK_SIZE = 16;

/// Number of elements per block (16 * 16 = 256)
constexpr std::size_t BSR_BLOCK_ELEMENTS = BSR_BLOCK_SIZE * BSR_BLOCK_SIZE;

/// Maximum supported matrix dimension (64K blocks = 1M rows/cols)
constexpr std::size_t BSR_MAX_DIMENSION = 65536 * BSR_BLOCK_SIZE;

//==============================================================================
// BSR Matrix Structure
//==============================================================================

/**
 * @brief BSR Matrix representation
 * 
 * Stores a sparse matrix in Block Sparse Row format where only non-zero
 * 16x16 blocks are stored.
 * 
 * Memory layout:
 * - data: Block values stored contiguously, row-major within each block
 * - col_idx: Column index for each non-zero block
 * - row_ptr: Cumulative count of non-zero blocks up to each row
 */
struct BSRMatrix {
    std::vector<std::int8_t> data;       ///< Block data (flattened, row-major within blocks)
    std::vector<std::size_t> col_idx;    ///< Column index for each non-zero block
    std::vector<std::size_t> row_ptr;    ///< Start index in col_idx for each block row
    std::size_t num_block_rows;          ///< Number of block rows
    std::size_t num_block_cols;          ///< Number of block columns
    std::size_t nnz_blocks;              ///< Number of non-zero blocks
    std::size_t orig_rows;               ///< Original matrix rows
    std::size_t orig_cols;               ///< Original matrix columns
    
    /// Default constructor
    BSRMatrix() : num_block_rows(0), num_block_cols(0), nnz_blocks(0),
                  orig_rows(0), orig_cols(0) {}
    
    /// Check if matrix is empty
    bool empty() const { return nnz_blocks == 0; }
    
    /// Get compression ratio (dense_size / bsr_size)
    float compression_ratio() const {
        if (data.empty()) return 0.0f;
        std::size_t dense_bytes = orig_rows * orig_cols;
        std::size_t bsr_bytes = data.size() + col_idx.size() * sizeof(std::size_t) 
                               + row_ptr.size() * sizeof(std::size_t);
        return static_cast<float>(dense_bytes) / static_cast<float>(bsr_bytes);
    }
};
//==============================================================================
// Validation Result Structure
//==============================================================================

/**
 * @brief Result of BSR validation
 */
struct ValidationResult {
    bool valid;              ///< True if BSR structure is valid
    std::string message;     ///< Error message if invalid
    std::size_t error_index; ///< Index where error was found (if applicable)
    
    ValidationResult() : valid(true), error_index(0) {}
    ValidationResult(bool v, const std::string& msg, std::size_t idx = 0) 
        : valid(v), message(msg), error_index(idx) {}
};

//============================================================================
// Core Block Scanning Function
//==============================================================================

/**
 * @brief Check if a block contains any non-zero element
 * 
 * Scans a block-sized region in the dense matrix to determine if it should
 * be stored in the BSR format (only non-zero blocks are stored).
 * 
 * @param block_start Pointer to start of block in dense matrix
 * @param block_row_stride Stride between rows in dense matrix (= cols)
 * @param block_rows Actual rows in this block (may be < 16 at edge)
 * @param block_cols Actual cols in this block (may be < 16 at edge)
 * @return true if block contains any non-zero element, false if all zeros
 */
inline bool is_block_nonzero(const std::int8_t* block_start, 
                              std::size_t block_row_stride,
                              std::size_t block_rows = BSR_BLOCK_SIZE,
                              std::size_t block_cols = BSR_BLOCK_SIZE) {
    for (std::size_t i = 0; i < block_rows; ++i) {
        for (std::size_t j = 0; j < block_cols; ++j) {
            if (block_start[i * block_row_stride + j] != 0) {
                return true;
            }
        }
    }
    return false;
}

//==============================================================================
// Core Pack Function
//==============================================================================

/**
 * @brief Convert a dense matrix to BSR format
 * 
 * Two-pass algorithm:
 * 1. First pass: Count non-zero blocks, build row_ptr
 * 2. Second pass: Copy non-zero block data, build col_idx
 * 
 * @param dense Pointer to dense matrix data (row-major)
 * @param rows Number of rows in dense matrix
 * @param cols Number of columns in dense matrix
 * @return BSRMatrix in BSR format
 */
inline BSRMatrix pack_to_bsr(const std::int8_t* dense, 
                              std::size_t rows, 
                              std::size_t cols) {
    BSRMatrix bsr;
    bsr.orig_rows = rows;
    bsr.orig_cols = cols;
    
    // Calculate number of blocks (ceiling division)
    bsr.num_block_rows = (rows + BSR_BLOCK_SIZE - 1) / BSR_BLOCK_SIZE;
    bsr.num_block_cols = (cols + BSR_BLOCK_SIZE - 1) / BSR_BLOCK_SIZE;
    
    //--------------------------------------------------------------------------
    // Pass 1: Count non-zero blocks
    //--------------------------------------------------------------------------
    bsr.row_ptr.resize(bsr.num_block_rows + 1);
    bsr.row_ptr[0] = 0;
    
    std::size_t nnz = 0;
    for (std::size_t br = 0; br < bsr.num_block_rows; ++br) {
        std::size_t row_start = br * BSR_BLOCK_SIZE;
        std::size_t block_rows = std::min(BSR_BLOCK_SIZE, rows - row_start);
        
        for (std::size_t bc = 0; bc < bsr.num_block_cols; ++bc) {
            std::size_t col_start = bc * BSR_BLOCK_SIZE;
            std::size_t block_cols = std::min(BSR_BLOCK_SIZE, cols - col_start);
            
            const std::int8_t* block_ptr = dense + row_start * cols + col_start;
            
            if (is_block_nonzero(block_ptr, cols, block_rows, block_cols)) {
                ++nnz;
            }
        }
        bsr.row_ptr[br + 1] = nnz;
    }
    
    bsr.nnz_blocks = nnz;
    bsr.col_idx.resize(nnz);
    bsr.data.resize(nnz * BSR_BLOCK_ELEMENTS);
    
    //--------------------------------------------------------------------------
    // Pass 2: Copy non-zero blocks with zero-padding for edges
    //--------------------------------------------------------------------------
    std::size_t block_idx = 0;
    for (std::size_t br = 0; br < bsr.num_block_rows; ++br) {
        std::size_t row_start = br * BSR_BLOCK_SIZE;
        std::size_t block_rows = std::min(BSR_BLOCK_SIZE, rows - row_start);
        
        for (std::size_t bc = 0; bc < bsr.num_block_cols; ++br) {
            std::size_t col_start = bc * BSR_BLOCK_SIZE;
            std::size_t block_cols = std::min(BSR_BLOCK_SIZE, cols - col_start);
            
            const std::int8_t* src = dense + row_start * cols + col_start;
            
            if (is_block_nonzero(src, cols, block_rows, block_cols)) {
                bsr.col_idx[block_idx] = bc;
                
                // Copy block with zero-padding for edge blocks
                std::int8_t* dst = &bsr.data[block_idx * BSR_BLOCK_ELEMENTS];
                for (std::size_t i = 0; i < BSR_BLOCK_SIZE; ++i) {
                    for (std::size_t j = 0; j < BSR_BLOCK_SIZE; ++j) {
                        if (i < block_rows && j < block_cols) {
                            dst[i * BSR_BLOCK_SIZE + j] = src[i * cols + j];
                        } else {
                            dst[i * BSR_BLOCK_SIZE + j] = 0;
                        
                    }
                }
            }
            ++block_idx;
        }
    }
    
    return bsr;
}

//==============================================================================
// Core Unpack Function
//==============================================================================

/**
 * @brief Unpack BSR matrix back to dense format
 * 
 * Reconstructs the original dense matrix from BSR representation.
 * 
 * @param bsr BSR matrix to unpack
 * @param dense Output dense matrix (must be pre-allocated)
 * @param rows Number of rows in output
 * @param cols Number of columns in output
 */
inline void unpack_from_bsr(const BSRMatrix& bsr, 
                             std::int8_t* dense,
                             std::size_t rows,
                             std::size_t cols) {
    // Zero the output first
    std::memset(dense, 0, rows * cols);
    
    // Copy each non-zero block back to dense matrix
    for (std::size_t br = 0; br < bsr.num_block_rows; ++br) {
        std::size_t row_start = br * BSR_BLOCK_SIZE;
        std::size_t block_rows = std::min(BSR_BLOCK_SIZE, rows - row_start);
        
        for (std::size_t idx = bsr.row_ptr[br]; idx < bsr.row_ptr[br + 1]; ++idx) {
            std::size_t bc = bsr.col_idx[idx];
            std::size_t col_start = bc * BSR_BLOCK_SIZE;
            std::size_t block_cols = std::min(BSR_BLOCK_SIZE, cols - col_start);
            
            const std::int8_t* src = &bsr.data[idx * BSR_BLOCK_ELEMENTS];
            std::int8_t* dst = dense + row_start * cols + col_start;
            
            // Copy block from BSR back to dense (skipping zero-padding)
            for (std::size_t i = 0; i < block_rows; ++i) {
                for (std::size_t j = 0; j < block_cols; ++j) {
                    dst[i * cols + j] = src[i * BSR_BLOCK_SIZE + j];
                }
            }
        }
    }
}

//==============================================================================
// Statistics Functions
//==============================================================================

/**
 * @brief Calculate block-level sparsity ratio
 * @return Fraction of blocks that are zero (0.0 to 1.0)
 */
inline float get_sparsity(const BSRMatrix& bsr) {
    std::size_t total_blocks = bsr.num_block_rows * bsr.num_block_cols;
    if (total_blocks == 0) return 0.0f;
    return 1.0f - static_cast<float>(bsr.nnz_blocks) / static_cast<float>(total_blocks);
}

/**
 * @brief Calculate element-level sparsity (zeros within non-zero blocks)
 */
inline float get_element_sparsity(const BSRMatrix& bsr) {
    if (bsr.data.empty()) return 0.0f;
    
    std::size_t zero_count = 0;
    for (std::int8_t val : bsr.data) {
        if (val == 0) ++zero_count;
    }
    return static_cast<float>(zero_count) / static_cast<float>(bsr.data.size());
}

/**
 * @brief Get memory usage of BSR representation in bytes
 */
inline std::size_t get_bsr_memory_bytes(const BSRMatrix& bsr) {
    return bsr.data.size() * sizeof(std::int8_t) +
           bsr.col_idx.size() * sizeof(std::size_t) +
           bsr.row_ptr.size() * sizeof(std::size_t);
}

/**
 * @brief Get memory usage of equivalent dense matrix in bytes
 */
inline std::size_t get_dense_memory_bytes(const BSRMatrix& bsr) {
    return bsr.orig_rows * bsr.orig_cols * sizeof(std::int8_t);
}

/**
 * @brief Get hardware-format memory usage (smaller indices)
 */
inline std::size_t get_hardware_memory_bytes(const BSRMatrix& bsr) {
    return 12 +  // Header
           (bsr.num_block_rows + 1) * 2 +  // row_ptr as uint16
           bsr.nnz_blocks * 2 +  // col_idx as uint16
           bsr.nnz_blocks * BSR_BLOCK_ELEMENTS;  // Block data
}

/**
 * @brief Get detailed statistics as a string
 */
inline std::string get_statistics(const BSRMatrix& bsr) {
    std::ostringstream ss;
    ss << "BSR Matrix Statistics:\n";
    ss << "  Original dimensions: " << bsr.orig_rows << " x " << bsr.orig_cols << "\n";
    ss << "  Block dimensions: " << bsr.num_block_rows << " x " << bsr.num_block_cols << "\n";
    ss << "  Total blocks: " << (bsr.num_block_rows * bsr.num_block_cols) << "\n";
    ss << "  Non-zero blocks: " << bsr.nnz_blocks << "\n";
    ss << "  Block sparsity: " << (get_sparsity(bsr) * 100.0f) << "%\n";
    ss << "  Element sparsity (in NNZ blocks): " << (get_element_sparsity(bsr) * 100.0f) << "%\n";
    ss << "  Dense memory: " << get_dense_memory_bytes(bsr) << " bytes\n";
    ss << "  BSR memory: " << get_bsr_memory_bytes(bsr) << " bytes\n";
    ss << "  Hardware memory: " << get_hardware_memory_bytes(bsr) << " bytes\n";
    ss << "  Compression ratio: " << bsr.compression_ratio() << "x\n";
    return ss.str();
}

//==============================================================================
// Validation Functions
//==============================================================================

/**
 * @brief Validate BSR matrix structure integrity
 * 
 * Checks:
 * 1. row_ptr is monotonically increasing
 * 2. row_ptr[0] == 0 and row_ptr[last] == nnz_blocks
 * 3. col_idx values are within bounds
 * 4. col_idx is sorted within each row
 * 5. Data array has correct size
 * 
 * @param bsr BSR matrix to validate
 * @return ValidationResult with status and error message
 */
inline ValidationResult validate_bsr(const BSRMatrix& bsr) {
    // Check row_ptr size
    if (bsr.row_ptr.size() != bsr.num_block_rows + 1) {
        return ValidationResult(false, 
            "row_ptr size mismatch: expected " + std::to_string(bsr.num_block_rows + 1) +
            ", got " + std::to_string(bsr.row_ptr.size()));
    }
    
    // Check row_ptr starts at 0
    if (!bsr.row_ptr.empty() && bsr.row_ptr[0] != 0) {
        return ValidationResult(false, 
            "row_ptr[0] should be 0, got " + std::to_string(bsr.row_ptr[0]), 0);
    }
    
    // Check row_ptr ends at nnz_blocks
    if (!bsr.row_ptr.empty() && bsr.row_ptr[bsr.num_block_rows] != bsr.nnz_blocks) {
        return ValidationResult(false,
            "row_ptr[last] should be " + std::to_string(bsr.nnz_blocks) +
            ", got " + std::to_string(bsr.row_ptr[bsr.num_block_rows]));
    }
    
    // Check row_ptr is monotonically increasing
    for (std::size_t i = 1; i <= bsr.num_block_rows; ++i) {
        if (bsr.row_ptr[i] < bsr.row_ptr[i - 1]) {
            return ValidationResult(false,
                "row_ptr not monotonically increasing at index " + std::to_string(i), i);
        }
    }
    
    // Check col_idx size
    if (bsr.col_idx.size() != bsr.nnz_blocks) {
        return ValidationResult(false,
            "col_idx size mismatch: expected " + std::to_string(bsr.nnz_blocks) +
            ", got " + std::to_string(bsr.col_idx.size()));
    }
    
    // Check col_idx values and sorting within each row
    for (std::size_t br = 0; br < bsr.num_block_rows; ++br) {
        std::size_t prev_col = 0;
        bool first = true;
        
        for (std::size_t idx = bsr.row_ptr[br]; idx < bsr.row_ptr[br + 1]; ++idx) {
            std::size_t col = bsr.col_idx[idx];
            
            // Check bounds
            if (col >= bsr.num_block_cols) {
                return ValidationResult(false,
                    "col_idx[" + std::to_string(idx) + "] = " + std::to_string(col) +
                    " exceeds num_block_cols = " + std::to_string(bsr.num_block_cols), idx);
            }
            
            // Check sorted within row
            if (!first && col <= prev_col) {
                return ValidationResult(false,
                    "col_idx not sorted in row " + std::to_string(br) +
                    " at index " + std::to_string(idx), idx);
            }
            
            prev_col = col;
            first = false;
        }
    }
    
    // Check data size
    if (bsr.data.size() != bsr.nnz_blocks * BSR_BLOCK_ELEMENTS) {
        return ValidationResult(false,
            "data size mismatch: expected " + 
            std::to_string(bsr.nnz_blocks * BSR_BLOCK_ELEMENTS) +
            ", got " + std::to_string(bsr.data.size()));
    }
    
    return ValidationResult(true, "Valid BSR structure");
}

/**
 * @brief Check if block contains all zeros (warning, not error)
 */
inline bool has_zero_blocks(const BSRMatrix& bsr) {
    for (std::size_t idx = 0; idx < bsr.nnz_blocks; ++idx) {
        const std::int8_t* block = &bsr.data[idx * BSR_BLOCK_ELEMENTS];
        bool has_nonzero = false;
        for (std::size_t i = 0; i < BSR_BLOCK_ELEMENTS; ++i) {
            if (block[i] != 0) {
                has_nonzero = true;
                break;
            }
        }
        if (!has_nonzero) {
            return true;  // Found an all-zero block
        }
    }
    return false;
}

/**
 * @brief Verify pack/unpack round-trip produces identical matrix
 */
inline bool verify_round_trip(const std::int8_t* original,
                               std::size_t rows, std::size_t cols) {
    // Pack
    BSRMatrix bsr = pack_to_bsr(original, rows, cols);
    
    // Unpack
    std::vector<std::int8_t> unpacked(rows * cols);
    unpack_from_bsr(bsr, unpacked.data(), rows, cols);
    
    // Compare
    return std::memcmp(original, unpacked.data(), rows * cols) == 0;
}

//==============================================================================
// Serialization Functions (Hardware Format)
//==============================================================================

/**
 * @brief Serialize BSR matrix to binary format for hardware DMA
 * 
 * Format:
 *   [4B] nnz_blocks (uint32)
 *   [4B] num_block_rows (uint32)
 *   [4B] num_block_cols (uint32)
 *   [2B * (num_block_rows + 1)] row_ptr (uint16[])
 *   [2B * nnz_blocks] col_idx (uint16[])
 *   [256B * nnz_blocks] block data (int8[])
 */
inline std::vector<std::uint8_t> serialize_for_hardware(const BSRMatrix& bsr) {
    // Calculate total size
    std::size_t header_size = 12;  // 3 x 4 bytes
    std::size_t row_ptr_size = (bsr.num_block_rows + 1) * 2;
    std::size_t col_idx_size = bsr.nnz_blocks * 2;
    std::size_t data_size = bsr.nnz_blocks * BSR_BLOCK_ELEMENTS;
    std::size_t total_size = header_size + row_ptr_size + col_idx_size + data_size;
    
    std::vector<std::uint8_t> buffer(total_size);
    std::size_t offset = 0;
    
    // Write header
    std::uint32_t nnz = static_cast<std::uint32_t>(bsr.nnz_blocks);
    std::uint32_t nbr = static_cast<std::uint32_t>(bsr.num_block_rows);
    std::uint32_t nbc = static_cast<std::uint32_t>(bsr.num_block_cols);
    
    std::memcpy(&buffer[offset], &nnz, 4); offset += 4;
    std::memcpy(&buffer[offset], &nbr, 4); offset += 4;
    std::memcpy(&buffer[offset], &nbc, 4); offset += 4;
    
    // Write row_ptr as uint16_t
    for (std::size_t i = 0; i <= bsr.num_block_rows; ++i) {
        std::uint16_t val = static_cast<std::uint16_t>(bsr.row_ptr[i]);
        std::memcpy(&buffer[offset], &val, 2); offset += 2;
    }
    
    // Write col_idx as uint16_t
    for (std::size_t i = 0; i < bsr.nnz_blocks; ++i) {
        std::uint16_t val = static_cast<std::uint16_t>(bsr.col_idx[i]);
        std::memcpy(&buffer[offset], &val, 2); offset += 2;
    }
    
    // Write block data
    std::memcpy(&buffer[offset], bsr.data.data(), bsr.data.size());
    
    return buffer;
}

/**
 * @brief Deserialize BSR matrix from hardware binary format
 */
inline BSRMatrix deserialize_from_hardware(const std::uint8_t* buffer, std::size_t size) {
    BSRMatrix bsr;
    std::size_t offset = 0;
    
    // Validate minimum size
    if (size < 12) {
        throw std::runtime_error("Buffer too small for BSR header");
    }
    
    // Read header
    std::uint32_t nnz, nbr, nbc;
    std::memcpy(&nnz, &buffer[offset], 4); offset += 4;
    std::memcpy(&nbr, &buffer[offset], 4); offset += 4;
    std::memcpy(&nbc, &buffer[offset], 4); offset += 4;
    
    bsr.nnz_blocks = nnz;
    bsr.num_block_rows = nbr;
    bsr.num_block_cols = nbc;
    
    // Validate expected size
    std::size_t expected_size = 12 + (nbr + 1) * 2 + nnz * 2 + nnz * BSR_BLOCK_ELEMENTS;
    if (size < expected_size) {
        throw std::runtime_error("Buffer size mismatch: expected " + 
                                  std::to_string(expected_size) + 
                                  ", got " + std::to_string(size));
    }
    
    // Read row_ptr
    bsr.row_ptr.resize(nbr + 1);
    for (std::size_t i = 0; i <= nbr; ++i) {
        std::uint16_t val;
        std::memcpy(&val, &buffer[offset], 2); offset += 2;
        bsr.row_ptr[i] = val;
    }
    
    // Read col_idx
    bsr.col_idx.resize(nnz);
    for (std::size_t i = 0; i < nnz; ++i) {
        std::uint16_t val;
        std::memcpy(&val, &buffer[offset], 2); offset += 2;
        bsr.col_idx[i] = val;
    }
    
    // Read block data
    bsr.data.resize(nnz * BSR_BLOCK_ELEMENTS);
    std::memcpy(bsr.data.data(), &buffer[offset], bsr.data.size());
    
    return bsr;
}

/**
 * @brief Verify serialization round-trip
 */
inline bool verify_serialization(const BSRMatrix& original) {
    auto bytes = serialize_for_hardware(original);
    auto restored = deserialize_from_hardware(bytes.data(), bytes.size());
    
    return original.nnz_blocks == restored.nnz_blocks &&
           original.num_block_rows == restored.num_block_rows &&
           original.num_block_cols == restored.num_block_cols &&
           original.row_ptr == restored.row_ptr &&
           original.col_idx == restored.col_idx &&
           original.data == restored.data;
}

//==============================================================================
// Unit Test Functions
//==============================================================================

/**
 * @brief Test pack/unpack with all-zeros matrix
 */
inline bool test_all_zeros() {
    std::vector<std::int8_t> dense(64 * 64, 0);
    BSRMatrix bsr = pack_to_bsr(dense.data(), 64, 64);
    
    // All zeros should produce empty BSR
    if (bsr.nnz_blocks != 0) return false;
    
    // Unpack should still work
    std::vector<std::int8_t> unpacked(64 * 64);
    unpack_from_bsr(bsr, unpacked.data(), 64, 64);
    
    // Should be all zeros
    for (auto v : unpacked) {
        if (v != 0) return false;
    }
    
    return true;
}

/**
 * @brief Test serialization round-trip (pack → serialize → deserialize)
 */
inline bool test_serialization() {
    std::vector<std::int8_t> dense(32 * 32, 0);
    dense[0] = 1;
    dense[17 * 32 + 17] = 2;
    
    BSRMatrix bsr = pack_to_bsr(dense.data(), 32, 32);
    return verify_serialization(bsr);
}

/**
 * @brief Test round-trip for non-aligned dimensions (17x17)
 */
inline bool test_non_aligned() {
    std::vector<std::int8_t> dense(17 * 17, 0);
    dense[0] = 1;
    dense[16 * 17 + 16] = 2;
    
    return verify_round_trip(dense.data(), 17, 17);
}

/**
 * @brief Run all critical unit tests
 * @return Number of failed tests (0 = all passed)
 */
inline int run_all_tests() {
    int failures = 0;
    
    std::cout << "Running BSR packer core tests...\n";
    
    if (!test_all_zeros()) {
        std::cout << "  FAIL: test_all_zeros\n";
        failures++;
    } else {
        std::cout << "  PASS: test_all_zeros\n";
    }
    
    if (!test_serialization()) {
        std::cout << "  FAIL: test_serialization\n";
        failures++;
    } else {
        std::cout << "  PASS: test_serialization\n";
    }
    
    if (!test_non_aligned()) {
        std::cout << "  FAIL: test_non_aligned\n";
        failures++;
    } else {
        std::cout << "  PASS: test_non_aligned\n";
    }
    
    std::cout << "\nResults: " << (3 - failures) << "/3 tests passed\n";
    
    return failures;
}

} // namespace resnet_accel

}

#endif // BSR_PACKER_HPP

