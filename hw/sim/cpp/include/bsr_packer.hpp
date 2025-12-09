/**
 * ╔═══════════════════════════════════════════════════════════════════════════╗
 * ║                          BSR_PACKER.HPP                                   ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║  REPLACES: sw/training/export_bsr.py, sw/exporters/bsr_exporter.py       ║
 * ╠═══════════════════════════════════════════════════════════════════════════╣
 * ║                                                                           ║
 * ║  PURPOSE:                                                                 ║
 * ║    Convert dense weight matrices to Block Sparse Row (BSR) format        ║
 * ║    with 16x16 blocks optimized for the systolic array. This is THE       ║
 * ║    key to making sparse acceleration work.                               ║
 * ║                                                                           ║
 * ║  WHY C++ INSTEAD OF PYTHON:                                               ║
 * ║    • 10-100x faster packing for large models (ResNet has millions        ║
 * ║      of weights)                                                          ║
 * ║    • Bit-exact output matching hardware expectations                      ║
 * ║    • Can pack on-the-fly during inference if needed                      ║
 * ║    • Same code runs in simulation and on ARM processor                    ║
 * ║                                                                           ║
 * ║  WHAT IS BSR FORMAT:                                                      ║
 * ║    Block Sparse Row stores only non-zero 16x16 blocks:                   ║
 * ║                                                                           ║
 * ║    Dense matrix (64x64):            BSR representation:                  ║
 * ║    ┌────┬────┬────┬────┐            row_ptr = [0, 2, 3, 5, 6]            ║
 * ║    │████│    │████│    │            col_idx = [0, 2, 1, 0, 2, 3]         ║
 * ║    ├────┼────┼────┼────┤            data = [B0, B2, B5, B8, B10, B15]    ║
 * ║    │    │████│    │    │                                                 ║
 * ║    ├────┼────┼────┼────┤            Where each B is a 16x16 block       ║
 * ║    │████│    │████│    │            (256 INT8 values = 256 bytes)        ║
 * ║    ├────┼────┼────┼────┤                                                 ║
 * ║    │    │    │    │████│            Sparsity = (16-6)/16 = 62.5%         ║
 * ║    └────┴────┴────┴────┘                                                 ║
 * ║    (█ = non-zero block)                                                  ║
 * ║                                                                           ║
 * ║  HARDWARE FORMAT (for DMA transfer):                                     ║
 * ║    ┌─────────────────────────────────────────────────────────────────┐   ║
 * ║    │ Header (12 bytes)                                               │   ║
 * ║    │   [4B] nnz_blocks                                               │   ║
 * ║    │   [4B] num_block_rows                                           │   ║
 * ║    │   [4B] num_block_cols                                           │   ║
 * ║    ├─────────────────────────────────────────────────────────────────┤   ║
 * ║    │ Row Pointers (2 * (num_block_rows + 1) bytes)                   │   ║
 * ║    │   uint16_t row_ptr[num_block_rows + 1]                          │   ║
 * ║    ├─────────────────────────────────────────────────────────────────┤   ║
 * ║    │ Column Indices (2 * nnz_blocks bytes)                           │   ║
 * ║    │   uint16_t col_idx[nnz_blocks]                                  │   ║
 * ║    ├─────────────────────────────────────────────────────────────────┤   ║
 * ║    │ Block Data (256 * nnz_blocks bytes)                             │   ║
 * ║    │   int8_t blocks[nnz_blocks][16][16]  (row-major within block)   │   ║
 * ║    └─────────────────────────────────────────────────────────────────┘   ║
 * ║                                                                           ║
 * ║  KEY DATA STRUCTURES:                                                     ║
 * ║                                                                           ║
 * ║    struct BSRMatrix {                                                     ║
 * ║        size_t num_block_rows;     // Rows / 16                           ║
 * ║        size_t num_block_cols;     // Cols / 16                           ║
 * ║        size_t nnz_blocks;         // Non-zero block count                ║
 * ║        vector<uint16_t> row_ptr;  // Cumulative block count per row      ║
 * ║        vector<uint16_t> col_idx;  // Column index of each block          ║
 * ║        vector<int8_t> data;       // Flattened block data                ║
 * ║    };                                                                     ║
 * ║                                                                           ║
 * ║  FUNCTIONS TO IMPLEMENT:                                                  ║
 * ║                                                                           ║
 * ║    BSRMatrix dense_to_bsr(dense, rows, cols, threshold)                  ║
 * ║      - Scan dense matrix for non-zero 16x16 blocks                       ║
 * ║      - Use Frobenius norm or max-abs to determine if block is "zero"     ║
 * ║      - Build row_ptr, col_idx, and data arrays                           ║
 * ║                                                                           ║
 * ║    void bsr_to_dense(bsr, dense, rows, cols)                             ║
 * ║      - Reconstruct dense matrix from BSR (for verification)              ║
 * ║                                                                           ║
 * ║    vector<uint8_t> pack_for_hardware(bsr)                                ║
 * ║      - Serialize BSRMatrix to byte stream for DMA                        ║
 * ║                                                                           ║
 * ║    BSRMatrix unpack_from_hardware(data)                                  ║
 * ║      - Deserialize from byte stream                                      ║
 * ║                                                                           ║
 * ║    bool validate_bsr(bsr)                                                ║
 * ║      - Check row_ptr is monotonic                                        ║
 * ║      - Check col_idx values in range                                     ║
 * ║      - Check data size matches nnz_blocks * 256                          ║
 * ║                                                                           ║
 * ║  THRESHOLD SELECTION:                                                     ║
 * ║    - Too low: Many tiny blocks kept, less speedup                        ║
 * ║    - Too high: Accuracy drops, model quality suffers                     ║
 * ║    - ResNet-18 typically uses threshold giving 50-70% block sparsity     ║
 * ║    - Use validation set to tune threshold per layer                      ║
 * ║                                                                           ║
 * ╚═══════════════════════════════════════════════════════════════════════════╝
 */

#ifndef BSR_PACKER_HPP
#define BSR_PACKER_HPP

#include <cstdint>
#include <vector>
#include <string>

// Block size must match systolic array dimensions
constexpr size_t BSR_BLOCK_SIZE = 16;
constexpr size_t BSR_BLOCK_ELEMENTS = BSR_BLOCK_SIZE * BSR_BLOCK_SIZE;  // 256

/**
 * Block Sparse Row matrix representation
 */
struct BSRMatrix {
    size_t num_block_rows;      // Original rows / 16 (rounded up)
    size_t num_block_cols;      // Original cols / 16 (rounded up)
    size_t nnz_blocks;          // Number of non-zero blocks
    
    std::vector<uint16_t> row_ptr;   // Size: num_block_rows + 1
    std::vector<uint16_t> col_idx;   // Size: nnz_blocks
    std::vector<int8_t> data;        // Size: nnz_blocks * 256
    
    // TODO: Implement these helper methods
    float sparsity() const;          // Percentage of zero blocks
    size_t memory_bytes() const;     // Total memory footprint
    size_t dense_bytes() const;      // Equivalent dense size
    float compression_ratio() const; // dense_bytes / memory_bytes
};

/**
 * BSR Packer class - handles all BSR conversions
 */
class BSRPacker {
public:
    BSRPacker() = default;
    
    /**
     * Convert dense matrix to BSR format
     * 
     * @param dense      Input dense matrix (row-major, INT8)
     * @param rows       Number of rows
     * @param cols       Number of columns
     * @param threshold  Block magnitude threshold (0.0 = keep all)
     * @return           BSRMatrix structure
     * 
     * TODO: Implement this - core algorithm:
     *   1. Pad rows/cols to multiples of 16
     *   2. For each 16x16 block, compute Frobenius norm
     *   3. If norm >= threshold, add to BSR structure
     *   4. Build row_ptr as cumulative sum
     */
    BSRMatrix dense_to_bsr(const int8_t* dense, size_t rows, size_t cols,
                           float threshold = 0.0f);
    
    /**
     * Convert BSR back to dense (for verification)
     * 
     * TODO: Implement this:
     *   1. Allocate output with zeros
     *   2. For each block, copy data to correct position
     */
    void bsr_to_dense(const BSRMatrix& bsr, int8_t* dense,
                      size_t rows, size_t cols);
    
    /**
     * Pack BSR matrix for hardware DMA transfer
     * 
     * TODO: Implement this - pack in exact hardware format:
     *   - Header: nnz_blocks, num_block_rows, num_block_cols
     *   - Row pointers array
     *   - Column indices array
     *   - Block data (aligned to 256 bytes each)
     */
    std::vector<uint8_t> pack_for_hardware(const BSRMatrix& bsr);
    
    /**
     * Unpack BSR matrix from hardware format
     */
    BSRMatrix unpack_from_hardware(const uint8_t* data, size_t size);
    
    /**
     * Validate BSR structure integrity
     * 
     * TODO: Implement checks:
     *   - row_ptr[0] == 0
     *   - row_ptr is monotonically non-decreasing
     *   - row_ptr[num_block_rows] == nnz_blocks
     *   - All col_idx values < num_block_cols
     *   - data.size() == nnz_blocks * 256
     */
    bool validate(const BSRMatrix& bsr);
    
    /**
     * Compute block Frobenius norm
     * sqrt(sum of squares of all elements)
     */
    static float block_frobenius_norm(const int8_t* block);
    
    /**
     * Compute block max absolute value
     * Alternative to Frobenius norm, faster
     */
    static int8_t block_max_abs(const int8_t* block);
    
    /**
     * Load BSR from numpy files (exported from Python training)
     * 
     * TODO: Implement .npy file parsing
     */
    BSRMatrix load_from_numpy(const std::string& weight_file,
                              const std::string& row_ptr_file,
                              const std::string& col_idx_file);
    
    /**
     * Save BSR to binary file
     */
    void save_to_file(const BSRMatrix& bsr, const std::string& filename);
    
    /**
     * Load BSR from binary file
     */
    BSRMatrix load_from_file(const std::string& filename);
};

#endif // BSR_PACKER_HPP
