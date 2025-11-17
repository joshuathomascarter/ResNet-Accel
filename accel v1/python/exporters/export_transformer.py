"""
Export Toy Transformer Q/K/V Weights with 80-90% Sparsity
Creates fixture data for hardware testing with realistic Transformer patterns.
"""

import os
import json
import numpy as np
from typing import Dict, Tuple

# Add parent to path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.export_bsr import build_bsr_from_dense, save_bsr_binary_int8, save_bsr_metadata


def create_sparse_mask(shape: Tuple[int, int], sparsity_pct: float, block_size: int = 8, seed: int = 42) -> np.ndarray:
    """
    Create block-sparse mask with specified sparsity percentage.

    Args:
        shape: Matrix shape (rows, cols)
        sparsity_pct: Target sparsity (0-100)
        block_size: Block dimension
        seed: Random seed

    Returns:
        Binary mask: 1 = keep, 0 = zero out
    """
    np.random.seed(seed)

    rows, cols = shape
    block_rows = (rows + block_size - 1) // block_size
    block_cols = (cols + block_size - 1) // block_size

    # Create block-level mask
    total_blocks = block_rows * block_cols
    num_zero_blocks = int(total_blocks * sparsity_pct / 100.0)

    block_mask = np.ones(total_blocks, dtype=bool)
    zero_indices = np.random.choice(total_blocks, size=num_zero_blocks, replace=False)
    block_mask[zero_indices] = False

    # Expand to element-level mask
    mask = np.zeros((rows, cols), dtype=np.float32)
    for idx, keep in enumerate(block_mask):
        if keep:
            block_row = idx // block_cols
            block_col = idx % block_cols

            r_start = block_row * block_size
            r_end = min(r_start + block_size, rows)
            c_start = block_col * block_size
            c_end = min(c_start + block_size, cols)

            mask[r_start:r_end, c_start:c_end] = 1.0

    return mask


def create_transformer_qkv_weights(
    d_model: int = 128, d_head: int = 64, sparsity_pct: float = 90.0, seed: int = 42
) -> Dict:
    """
    Create toy Transformer Q/K/V weight matrices with structured sparsity.

    Args:
        d_model: Model dimension
        d_head: Attention head dimension
        sparsity_pct: Target sparsity percentage
        seed: Random seed

    Returns:
        Dictionary with Q, K, V weights and metadata
    """
    np.random.seed(seed)

    # Initialize weights with Xavier initialization
    scale = np.sqrt(2.0 / (d_model + d_head))

    Q = np.random.randn(d_model, d_head).astype(np.float32) * scale
    K = np.random.randn(d_model, d_head).astype(np.float32) * scale
    V = np.random.randn(d_model, d_head).astype(np.float32) * scale

    # Apply block-sparse masks
    Q_mask = create_sparse_mask(Q.shape, sparsity_pct, block_size=8, seed=seed)
    K_mask = create_sparse_mask(K.shape, sparsity_pct, block_size=8, seed=seed + 1)
    V_mask = create_sparse_mask(V.shape, sparsity_pct, block_size=8, seed=seed + 2)

    Q_sparse = Q * Q_mask
    K_sparse = K * K_mask
    V_sparse = V * V_mask

    # Compute actual sparsity
    Q_sparsity = 100.0 * (1.0 - np.count_nonzero(Q_sparse) / Q_sparse.size)
    K_sparsity = 100.0 * (1.0 - np.count_nonzero(K_sparse) / K_sparse.size)
    V_sparsity = 100.0 * (1.0 - np.count_nonzero(V_sparse) / V_sparse.size)

    return {
        "Q": Q_sparse,
        "K": K_sparse,
        "V": V_sparse,
        "metadata": {
            "d_model": d_model,
            "d_head": d_head,
            "target_sparsity": sparsity_pct,
            "actual_sparsity": {"Q": float(Q_sparsity), "K": float(K_sparsity), "V": float(V_sparsity)},
            "block_size": 8,
        },
    }


def export_transformer_fixtures(output_dir: str = None):
    """Export Transformer Q/K/V weights at 80% and 90% sparsity"""

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "../../data/fixtures/transformer")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Exporting Transformer Q/K/V Fixtures")
    print("=" * 60)

    for sparsity in [80.0, 90.0]:
        print(f"\nGenerating {sparsity}% sparse fixtures...")

        qkv = create_transformer_qkv_weights(d_model=128, d_head=64, sparsity_pct=sparsity)

        # Save dense NumPy arrays
        filename = f"qkv_weights_{int(sparsity)}pct_sparse.npz"
        filepath = os.path.join(output_dir, filename)
        np.savez_compressed(filepath, Q=qkv["Q"], K=qkv["K"], V=qkv["V"])
        print(f"  Saved dense weights: {filename}")

        # Build BSR format for each matrix
        for name, weight in [("Q", qkv["Q"]), ("K", qkv["K"]), ("V", qkv["V"])]:
            bsr = build_bsr_from_dense(weight, 8, 8)

            # Save BSR binary
            bsr_dir = os.path.join(output_dir, f"{int(sparsity)}pct", name.lower())
            os.makedirs(bsr_dir, exist_ok=True)

            # Quantize to INT8
            scales = np.max(np.abs(weight), axis=1, keepdims=True) / 127.0
            scales = np.maximum(scales, 1e-12).flatten()

            # Save INT8 BSR
            bsr_file = os.path.join(bsr_dir, "weights_int8.bsr")
            meta_file = os.path.join(bsr_dir, "weights.meta.json")
            scales_file = os.path.join(bsr_dir, "scales.npy")

            save_bsr_binary_int8(bsr, scales, bsr_file)
            save_bsr_metadata(bsr, meta_file, layer_name=name)
            np.save(scales_file, scales)

            print(f"    {name}: {bsr['num_blocks']} blocks, {bsr['sparsity_pct']:.1f}% sparse")

        # Save metadata
        meta_file = os.path.join(output_dir, f"metadata_{int(sparsity)}pct.json")
        with open(meta_file, "w") as f:
            json.dump(qkv["metadata"], f, indent=2)

    print("\n" + "=" * 60)
    print("Transformer fixtures exported successfully!")
    print(f"Location: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    export_transformer_fixtures()
