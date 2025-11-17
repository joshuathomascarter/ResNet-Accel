"""
Export MLP (Fully Connected) Layer Fixtures
Creates sparse FC layer weights for hardware testing.
"""

import os
import json
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.export_bsr import build_bsr_from_dense, save_bsr_binary_int8, save_bsr_metadata
from exporters.export_transformer import create_sparse_mask


def create_mlp_weights(input_dim: int = 512, output_dim: int = 128, sparsity_pct: float = 90.0, seed: int = 42) -> dict:
    """Create sparse MLP layer weights"""
    np.random.seed(seed)

    # Xavier initialization
    scale = np.sqrt(2.0 / (input_dim + output_dim))
    weights = np.random.randn(output_dim, input_dim).astype(np.float32) * scale
    bias = np.zeros(output_dim, dtype=np.float32)

    # Apply block sparsity
    mask = create_sparse_mask(weights.shape, sparsity_pct, block_size=8, seed=seed)
    weights_sparse = weights * mask

    actual_sparsity = 100.0 * (1.0 - np.count_nonzero(weights_sparse) / weights_sparse.size)

    return {
        "weights": weights_sparse,
        "bias": bias,
        "metadata": {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "target_sparsity": sparsity_pct,
            "actual_sparsity": float(actual_sparsity),
            "block_size": 8,
        },
    }


def export_mlp_fixtures(output_dir: str = None):
    """Export MLP fixtures at various sparsity levels"""

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "../../data/fixtures/mlp")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Exporting MLP Fixtures")
    print("=" * 60)

    configs = [
        {"input_dim": 512, "output_dim": 128, "sparsity": 90.0},
        {"input_dim": 1024, "output_dim": 256, "sparsity": 85.0},
        {"input_dim": 9216, "output_dim": 128, "sparsity": 90.0},  # MNIST FC1
    ]

    for cfg in configs:
        print(f"\nGenerating {cfg['input_dim']}→{cfg['output_dim']} @ {cfg['sparsity']}% sparse...")

        mlp = create_mlp_weights(cfg["input_dim"], cfg["output_dim"], cfg["sparsity"])

        # Save dense
        name = f"fc_{cfg['input_dim']}_{cfg['output_dim']}_{int(cfg['sparsity'])}pct.npz"
        filepath = os.path.join(output_dir, name)
        np.savez_compressed(filepath, weights=mlp["weights"], bias=mlp["bias"])

        # Build and save BSR
        bsr = build_bsr_from_dense(mlp["weights"], 8, 8)

        bsr_dir = os.path.join(output_dir, f"fc_{cfg['input_dim']}_{cfg['output_dim']}")
        os.makedirs(bsr_dir, exist_ok=True)

        # Quantize
        scales = np.max(np.abs(mlp["weights"]), axis=1, keepdims=True) / 127.0
        scales = np.maximum(scales, 1e-12).flatten()

        save_bsr_binary_int8(bsr, scales, os.path.join(bsr_dir, "weights_int8.bsr"))
        save_bsr_metadata(
            bsr, os.path.join(bsr_dir, "weights.meta.json"), layer_name=f"fc_{cfg['input_dim']}_{cfg['output_dim']}"
        )
        np.save(os.path.join(bsr_dir, "scales.npy"), scales)
        np.save(os.path.join(bsr_dir, "bias.npy"), mlp["bias"])

        print(f"  Blocks: {bsr['num_blocks']}, Sparsity: {bsr['sparsity_pct']:.1f}%")

        # Save metadata
        with open(os.path.join(bsr_dir, "metadata.json"), "w") as f:
            json.dump(mlp["metadata"], f, indent=2)

    print("\n" + "=" * 60)
    print("MLP fixtures exported!")
    print(f"Location: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    export_mlp_fixtures()
