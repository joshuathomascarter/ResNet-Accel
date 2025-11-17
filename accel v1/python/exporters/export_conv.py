"""
Export Convolutional Layer Fixtures
Creates sparse conv kernels (4×4 blocks) for hardware testing.
"""

import os
import json
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.export_bsr import build_bsr_from_dense, save_bsr_binary_int8, save_bsr_metadata
from exporters.export_transformer import create_sparse_mask


def create_conv_weights(
    in_channels: int = 32, out_channels: int = 64, kernel_size: int = 3, sparsity_pct: float = 70.0, seed: int = 42
) -> dict:
    """Create sparse convolutional kernel weights"""
    np.random.seed(seed)

    # Conv weights: [out_channels, in_channels, kernel_h, kernel_w]
    # Flatten to 2D for BSR: [out_channels, in_channels * kernel_h * kernel_w]
    scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
    weights_4d = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * scale

    # Flatten to 2D for block sparsity
    weights_2d = weights_4d.reshape(out_channels, -1)

    # Apply block sparsity (4×4 for conv)
    mask = create_sparse_mask(weights_2d.shape, sparsity_pct, block_size=4, seed=seed)
    weights_sparse_2d = weights_2d * mask

    # Reshape back to 4D
    weights_sparse_4d = weights_sparse_2d.reshape(out_channels, in_channels, kernel_size, kernel_size)

    bias = np.zeros(out_channels, dtype=np.float32)

    actual_sparsity = 100.0 * (1.0 - np.count_nonzero(weights_sparse_2d) / weights_sparse_2d.size)

    return {
        "weights_4d": weights_sparse_4d,
        "weights_2d": weights_sparse_2d,
        "bias": bias,
        "metadata": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "target_sparsity": sparsity_pct,
            "actual_sparsity": float(actual_sparsity),
            "block_size": 4,
        },
    }


def export_conv_fixtures(output_dir: str = None):
    """Export Conv fixtures at various sparsity levels"""

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "../../data/fixtures/conv")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Exporting Conv Fixtures")
    print("=" * 60)

    configs = [
        {"in_ch": 32, "out_ch": 64, "k": 3, "sparsity": 70.0},
        {"in_ch": 64, "out_ch": 128, "k": 3, "sparsity": 75.0},
        {"in_ch": 1, "out_ch": 32, "k": 3, "sparsity": 50.0},  # MNIST conv1
    ]

    for cfg in configs:
        print(f"\nGenerating Conv {cfg['in_ch']}→{cfg['out_ch']}, k={cfg['k']}, {cfg['sparsity']}% sparse...")

        conv = create_conv_weights(cfg["in_ch"], cfg["out_ch"], cfg["k"], cfg["sparsity"])

        # Save dense
        name = f"conv_{cfg['in_ch']}_{cfg['out_ch']}_k{cfg['k']}_{int(cfg['sparsity'])}pct.npz"
        filepath = os.path.join(output_dir, name)
        np.savez_compressed(filepath, weights=conv["weights_4d"], bias=conv["bias"])

        # Build and save BSR (use 2D flattened version with 4×4 blocks)
        bsr = build_bsr_from_dense(conv["weights_2d"], 4, 4)

        bsr_dir = os.path.join(output_dir, f"conv_{cfg['in_ch']}_{cfg['out_ch']}_k{cfg['k']}")
        os.makedirs(bsr_dir, exist_ok=True)

        # Quantize
        scales = np.max(np.abs(conv["weights_2d"]), axis=1, keepdims=True) / 127.0
        scales = np.maximum(scales, 1e-12).flatten()

        save_bsr_binary_int8(bsr, scales, os.path.join(bsr_dir, "weights_int8.bsr"))
        save_bsr_metadata(
            bsr, os.path.join(bsr_dir, "weights.meta.json"), layer_name=f"conv_{cfg['in_ch']}_{cfg['out_ch']}"
        )
        np.save(os.path.join(bsr_dir, "scales.npy"), scales)
        np.save(os.path.join(bsr_dir, "bias.npy"), conv["bias"])

        print(f"  Blocks (4×4): {bsr['num_blocks']}, Sparsity: {bsr['sparsity_pct']:.1f}%")

        # Save metadata
        with open(os.path.join(bsr_dir, "metadata.json"), "w") as f:
            json.dump(conv["metadata"], f, indent=2)

    print("\n" + "=" * 60)
    print("Conv fixtures exported!")
    print(f"Location: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    export_conv_fixtures()
