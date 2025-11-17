"""
Tests for BSR exporter functions
Validates Transformer, MLP, and Conv fixture generation
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exporters.export_transformer import create_sparse_mask, create_transformer_qkv_weights
from exporters.export_mlp import create_mlp_weights
from exporters.export_conv import create_conv_weights


class TestSparseMask:
    def test_creates_correct_sparsity(self):
        """Verify sparsity percentage is approximately correct"""
        shape = (128, 128)
        sparsity = 80.0
        mask = create_sparse_mask(shape, sparsity, block_size=8, seed=42)

        actual_sparsity = 100.0 * (1.0 - np.count_nonzero(mask) / mask.size)
        assert abs(actual_sparsity - sparsity) < 5.0, f"Sparsity {actual_sparsity} not close to {sparsity}"

    def test_block_aligned(self):
        """Verify mask is block-aligned"""
        shape = (64, 64)
        mask = create_sparse_mask(shape, 75.0, block_size=8, seed=0)

        # Check all 8×8 blocks are either all-zero or all-one
        for i in range(0, 64, 8):
            for j in range(0, 64, 8):
                block = mask[i : i + 8, j : j + 8]
                assert np.all(block == 0) or np.all(block == 1), f"Block at ({i}, {j}) is not uniform"

    def test_reproducible(self):
        """Verify same seed gives same mask"""
        shape = (128, 128)
        mask1 = create_sparse_mask(shape, 80.0, block_size=8, seed=999)
        mask2 = create_sparse_mask(shape, 80.0, block_size=8, seed=999)

        assert np.array_equal(mask1, mask2), "Same seed should produce same mask"


class TestTransformerExport:
    def test_creates_qkv_weights(self):
        """Verify Transformer Q/K/V generation"""
        qkv = create_transformer_qkv_weights(d_model=512, d_head=64, sparsity_pct=85.0, seed=42)

        assert qkv["Q"].shape == (512, 64)
        assert qkv["K"].shape == (512, 64)
        assert qkv["V"].shape == (512, 64)

        # Check sparsity is within range
        for key in ["Q", "K", "V"]:
            sparsity = 100.0 * (1.0 - np.count_nonzero(qkv[key]) / qkv[key].size)
            assert 80.0 <= sparsity <= 90.0, f"{key} sparsity {sparsity} out of range"

    def test_metadata_correct(self):
        """Verify metadata has correct fields"""
        qkv = create_transformer_qkv_weights(d_model=512, d_head=64, sparsity_pct=85.0)

        assert "d_model" in qkv["metadata"]
        assert "d_head" in qkv["metadata"]
        assert qkv["metadata"]["block_size"] == 8


class TestMLPExport:
    def test_creates_mlp_weights(self):
        """Verify MLP weight generation"""
        mlp = create_mlp_weights(input_dim=512, output_dim=128, sparsity_pct=85.0, seed=42)

        assert mlp["weights"].shape == (128, 512)
        assert mlp["bias"].shape == (128,)

        sparsity = 100.0 * (1.0 - np.count_nonzero(mlp["weights"]) / mlp["weights"].size)
        assert 80.0 <= sparsity <= 90.0, f"Sparsity {sparsity} out of range"

    def test_mnist_fc1_size(self):
        """Verify MNIST FC1 dimensions (9216 → 128)"""
        mlp = create_mlp_weights(input_dim=9216, output_dim=128, sparsity_pct=85.0)

        assert mlp["weights"].shape == (128, 9216)
        assert mlp["metadata"]["input_dim"] == 9216
        assert mlp["metadata"]["output_dim"] == 128


class TestConvExport:
    def test_creates_conv_weights(self):
        """Verify Conv weight generation"""
        conv = create_conv_weights(in_channels=32, out_channels=64, kernel_size=3, sparsity_pct=70.0)

        assert conv["weights_4d"].shape == (64, 32, 3, 3)
        assert conv["bias"].shape == (64,)

        # Check sparsity on flattened version
        sparsity = conv["metadata"]["actual_sparsity"]
        assert 65.0 <= sparsity <= 75.0, f"Sparsity {sparsity} out of range"

    def test_block_size_4(self):
        """Verify Conv uses 4×4 blocks"""
        conv = create_conv_weights(in_channels=32, out_channels=64, kernel_size=3)

        assert conv["metadata"]["block_size"] == 4


class TestEdgeCases:
    def test_zero_sparsity(self):
        """Verify 0% sparsity creates dense matrix"""
        mask = create_sparse_mask((128, 128), sparsity_pct=0.0, block_size=8)

        assert np.all(mask == 1), "0% sparsity should be all ones"

    def test_hundred_percent_sparsity(self):
        """Verify 100% sparsity creates empty matrix"""
        mask = create_sparse_mask((128, 128), sparsity_pct=100.0, block_size=8)

        assert np.all(mask == 0), "100% sparsity should be all zeros"

    def test_non_divisible_shape(self):
        """Verify handling of shapes not divisible by block size"""
        # 130×130 not divisible by 8
        # Current implementation doesn't pad (works with partial blocks)
        mask = create_sparse_mask((130, 130), sparsity_pct=75.0, block_size=8)

        # Should maintain original shape
        assert mask.shape == (130, 130)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
