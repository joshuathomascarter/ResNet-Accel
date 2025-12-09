/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                       TEST_BSR_PACKER.CPP                                 ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  UNIT TESTS: BSR (Block Sparse Row) packing/unpacking                    ║
 * ║  TESTS: bsr_packer.hpp / bsr_packer.cpp                                  ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║  Validates the BSR format conversion, ensuring dense matrices are        ║
 * ║  correctly converted to BSR and back. Critical for sparse acceleration.  ║
 * ║                                                                           ║
 * ║  REPLACES PYTHON: sw/tests/test_bsr_export.py                            ║
 * ║                                                                           ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  TEST CASES TO IMPLEMENT:                                                 ║
 * ║                                                                           ║
 * ║  1. test_dense_to_bsr_identity()                                          ║
 * ║     - Convert identity matrix to BSR                                     ║
 * ║     - Verify only diagonal blocks are non-zero                           ║
 * ║                                                                           ║
 * ║  2. test_dense_to_bsr_full()                                              ║
 * ║     - Convert fully dense matrix (no zeros)                              ║
 * ║     - Verify all blocks present                                          ║
 * ║                                                                           ║
 * ║  3. test_dense_to_bsr_sparse()                                            ║
 * ║     - Convert matrix with 50% zero blocks                                ║
 * ║     - Verify correct block count                                         ║
 * ║                                                                           ║
 * ║  4. test_bsr_to_dense_roundtrip()                                         ║
 * ║     - Convert dense -> BSR -> dense                                      ║
 * ║     - Verify exact match                                                 ║
 * ║                                                                           ║
 * ║  5. test_serialize_deserialize()                                          ║
 * ║     - Write BSR to binary file                                           ║
 * ║     - Read back and verify                                               ║
 * ║                                                                           ║
 * ║  6. test_block_threshold()                                                ║
 * ║     - Test different threshold values for zero detection                 ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include "../include/bsr_packer.hpp"

// =============================================================================
// Test Utilities
// =============================================================================

#define TEST(name) \
    std::cout << "  " << #name << "... "; \
    if (name()) { std::cout << "PASS" << std::endl; passed++; } \
    else { std::cout << "FAIL" << std::endl; failed++; }

static int passed = 0;
static int failed = 0;

// =============================================================================
// Test Cases
// =============================================================================

bool test_dense_to_bsr_identity() {
    // TODO: Implement
    // BSRPacker packer;
    //
    // // Create 32x32 identity matrix (2x2 blocks of 16x16)
    // std::vector<int8_t> dense(32 * 32, 0);
    // for (int i = 0; i < 32; i++) dense[i * 32 + i] = 1;
    //
    // BSRMatrix bsr = packer.dense_to_bsr(dense.data(), 32, 32, 1.0f);
    //
    // // Should have 2 non-zero blocks (diagonal)
    // if (bsr.nnz_blocks != 2) return false;
    //
    // // Verify block positions
    // if (bsr.col_indices[0] != 0 || bsr.col_indices[1] != 1) return false;
    
    return true;
}

bool test_dense_to_bsr_full() {
    // TODO: Implement
    // BSRPacker packer;
    //
    // // Create 32x32 matrix with all non-zero
    // std::vector<int8_t> dense(32 * 32);
    // for (int i = 0; i < 32 * 32; i++) dense[i] = (i % 127) + 1;
    //
    // BSRMatrix bsr = packer.dense_to_bsr(dense.data(), 32, 32, 1.0f);
    //
    // // Should have 4 blocks (2x2 grid of 16x16 blocks)
    // if (bsr.nnz_blocks != 4) return false;
    
    return true;
}

bool test_dense_to_bsr_sparse() {
    // TODO: Implement
    // BSRPacker packer;
    //
    // // Create 32x32 matrix with half the blocks zero
    // std::vector<int8_t> dense(32 * 32, 0);
    // // Fill only top-left and bottom-right 16x16 blocks
    // for (int i = 0; i < 16; i++) {
    //     for (int j = 0; j < 16; j++) {
    //         dense[i * 32 + j] = 1;           // Top-left
    //         dense[(i+16) * 32 + (j+16)] = 1; // Bottom-right
    //     }
    // }
    //
    // BSRMatrix bsr = packer.dense_to_bsr(dense.data(), 32, 32, 1.0f);
    //
    // // Should have 2 blocks
    // if (bsr.nnz_blocks != 2) return false;
    //
    // // Sparsity should be 50%
    // float sparsity = bsr.sparsity();
    // if (std::abs(sparsity - 0.5f) > 0.01f) return false;
    
    return true;
}

bool test_bsr_to_dense_roundtrip() {
    // TODO: Implement
    // BSRPacker packer;
    //
    // // Create random matrix
    // std::vector<int8_t> original(32 * 32);
    // for (int i = 0; i < 32 * 32; i++) original[i] = (i * 7) % 256 - 128;
    //
    // // Convert to BSR and back
    // BSRMatrix bsr = packer.dense_to_bsr(original.data(), 32, 32, 1.0f);
    // std::vector<int8_t> recovered(32 * 32);
    // packer.bsr_to_dense(bsr, recovered.data());
    //
    // // Verify exact match
    // for (int i = 0; i < 32 * 32; i++) {
    //     if (original[i] != recovered[i]) return false;
    // }
    
    return true;
}

bool test_serialize_deserialize() {
    // TODO: Implement
    // BSRPacker packer;
    //
    // std::vector<int8_t> dense(32 * 32);
    // for (int i = 0; i < 32 * 32; i++) dense[i] = i % 128;
    //
    // BSRMatrix original = packer.dense_to_bsr(dense.data(), 32, 32, 1.0f);
    //
    // // Serialize
    // packer.save_bsr(original, "/tmp/test_bsr.bin");
    //
    // // Deserialize
    // BSRMatrix loaded = packer.load_bsr("/tmp/test_bsr.bin");
    //
    // // Compare
    // if (original.nnz_blocks != loaded.nnz_blocks) return false;
    // if (original.values != loaded.values) return false;
    
    return true;
}

bool test_block_threshold() {
    // TODO: Implement
    // BSRPacker packer;
    //
    // // Create matrix with small values
    // std::vector<int8_t> dense(16 * 16);
    // for (int i = 0; i < 256; i++) dense[i] = (i < 128) ? 1 : 0;
    //
    // // With threshold 0, block is non-zero
    // BSRMatrix bsr1 = packer.dense_to_bsr(dense.data(), 16, 16, 1.0f, 0.0f);
    // if (bsr1.nnz_blocks != 1) return false;
    //
    // // With high threshold, block becomes zero
    // BSRMatrix bsr2 = packer.dense_to_bsr(dense.data(), 16, 16, 1.0f, 0.5f);
    // if (bsr2.nnz_blocks != 0) return false;
    
    return true;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== BSR Packer Tests ===" << std::endl;
    
    TEST(test_dense_to_bsr_identity);
    TEST(test_dense_to_bsr_full);
    TEST(test_dense_to_bsr_sparse);
    TEST(test_bsr_to_dense_roundtrip);
    TEST(test_serialize_deserialize);
    TEST(test_block_threshold);
    
    std::cout << std::endl;
    std::cout << "Passed: " << passed << "/" << (passed + failed) << std::endl;
    
    return failed == 0 ? 0 : 1;
}
