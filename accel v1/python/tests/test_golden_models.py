"""
Tests for Golden Model GEMM implementations
Validates INT8 BSR GEMM functionality
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "golden")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from golden.gemm_bsr_int8 import gemm_bsr_int8
from training.export_bsr import build_bsr_from_dense


def quantize_per_channel(W):
    """Simple per-channel quantization"""
    scales = np.max(np.abs(W), axis=1, keepdims=True) / 127.0
    scales = np.maximum(scales, 1e-12)
    W_q = np.clip(np.round(W / scales), -128, 127).astype(np.int8)
    return W_q, scales.flatten()


class TestBSRInt8GEMM:
    def test_dense_bsr_gemm(self):
        """Verify BSR GEMM produces reasonable output with dense matrix"""
        M, K, N = 4, 64, 8
        A_fp32 = np.random.randn(M, K).astype(np.float32) * 0.1
        B_fp32 = np.random.randn(K, N).astype(np.float32) * 0.1

        # Quantize A
        scale_A = np.max(np.abs(A_fp32)) / 127.0
        A_int8 = np.clip(np.round(A_fp32 / scale_A), -128, 127).astype(np.int8)

        # Quantize B and build BSR
        B_int8, scales_B = quantize_per_channel(B_fp32.T)
        B_int8_T = B_int8.T

        bsr = build_bsr_from_dense(B_int8_T, 8, 8)

        # Run BSR GEMM
        C_bsr = gemm_bsr_int8(A_int8, bsr, scale_A, scales_B)

        # Should produce non-zero output with correct shape
        assert C_bsr.shape == (M, N), f"Output shape {C_bsr.shape} != ({M}, {N})"
        assert np.any(C_bsr != 0), "Output is all zeros"

    def test_sparse_bsr_gemm(self):
        """Verify BSR GEMM handles 50% sparsity"""
        M, K, N = 4, 64, 8
        A_fp32 = np.random.randn(M, K).astype(np.float32) * 0.1
        B_fp32 = np.random.randn(K, N).astype(np.float32) * 0.1

        # Make B sparse (zero out half the blocks)
        for i in range(0, K, 8):
            if i // 8 % 2 == 0:
                B_fp32[i : i + 8, :] = 0

        # Quantize
        scale_A = np.max(np.abs(A_fp32)) / 127.0
        A_int8 = np.clip(np.round(A_fp32 / scale_A), -128, 127).astype(np.int8)

        B_int8, scales_B = quantize_per_channel(B_fp32.T)
        B_int8_T = B_int8.T

        bsr = build_bsr_from_dense(B_int8_T, 8, 8)

        # Run BSR GEMM
        C_bsr = gemm_bsr_int8(A_int8, bsr, scale_A, scales_B)

        # Output should have reasonable shape
        assert C_bsr.shape == (M, N)

        # Should not be all zeros (some blocks are non-zero)
        assert np.any(C_bsr != 0)

    def test_all_zero_column(self):
        """Verify BSR GEMM handles completely zero columns"""
        M, K, N = 4, 64, 8
        A_fp32 = np.ones((M, K), dtype=np.float32) * 0.1
        B_fp32 = np.random.randn(K, N).astype(np.float32) * 0.1

        # Make entire first column zero
        B_fp32[:, 0] = 0

        # Quantize
        scale_A = np.max(np.abs(A_fp32)) / 127.0
        A_int8 = np.clip(np.round(A_fp32 / scale_A), -128, 127).astype(np.int8)

        # Handle zero column (scale will be zero, need protection)
        scales_per_col = np.max(np.abs(B_fp32), axis=0) / 127.0
        scales_per_col = np.maximum(scales_per_col, 1e-12)  # Avoid division by zero
        B_int8 = np.clip(np.round(B_fp32 / scales_per_col), -128, 127).astype(np.int8)

        bsr = build_bsr_from_dense(B_int8, 8, 8)

        # Run BSR GEMM
        C_bsr = gemm_bsr_int8(A_int8, bsr, scale_A, scales_per_col)

        # Output shape should be correct
        assert C_bsr.shape == (M, N)


class TestBSRStructure:
    def test_row_ptr_indexing(self):
        """Verify indptr correctly indexes blocks"""
        B = np.random.randn(64, 8).astype(np.float32)

        # Make first 8 rows zero (first block-row)
        B[0:8, :] = 0

        bsr = build_bsr_from_dense(B, 8, 8)

        # First indptr should have no blocks
        assert bsr["indptr"][1] - bsr["indptr"][0] == 0, "Empty block-row should have indptr[i+1] == indptr[i]"

    def test_block_count(self):
        """Verify num_blocks matches data array"""
        B = np.random.randn(64, 8).astype(np.float32)

        # Zero out half
        B[:32, :] = 0

        bsr = build_bsr_from_dense(B, 8, 8)

        expected_blocks = bsr["data"].shape[0]
        assert bsr["num_blocks"] == expected_blocks, f"num_blocks {bsr['num_blocks']} != {expected_blocks}"

    def test_sparsity_calculation(self):
        """Verify sparsity percentage calculation"""
        B = np.zeros((64, 8), dtype=np.float32)

        # Fill 25% of blocks
        B[0:8, 0:8] = np.random.randn(8, 8)
        B[16:24, 0:8] = np.random.randn(8, 8)

        bsr = build_bsr_from_dense(B, 8, 8)

        # Should report ~75% sparse
        assert 70.0 <= bsr["sparsity_pct"] <= 80.0, f"Sparsity {bsr['sparsity_pct']} out of expected range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
