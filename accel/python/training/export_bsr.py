"""
BSR Export for ACCEL-BSR Hardware
Exports block-sparse weights in hardware-ready BSR format with metadata.

Output format:
- *.bsr: Binary payload with non-zero blocks
- *.meta.json: Metadata with row_ptr, col_idx, block dimensions
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from typing import Dict, Tuple, List, Optional
import struct


# ----------------------------
# Configuration
# ----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SPARSE_MODEL_PATH = os.path.join(ROOT, "python", "training", "mnist_sparse_90pct.npz")
OUTPUT_DIR = os.path.join(ROOT, "data", "bsr_export")


# ----------------------------
# Model Definition (must match training)
# ----------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def layer_block_cfg(name, module):
    """Return block size and minimum keep percentage for each layer"""
    if isinstance(module, nn.Conv2d):
        return (4, 4), 0.30  # 4x4 blocks
    else:  # Linear
        return (8, 8), 0.05  # 8x8 blocks


# ----------------------------
# BSR Extraction Functions
# ----------------------------
def extract_bsr_from_scipy(bsr_matrix: sp.bsr_matrix) -> Dict:
    """
    Extract BSR components from scipy sparse matrix.

    Returns:
        Dictionary with data, indices, indptr, and block info
    """
    return {
        "data": bsr_matrix.data,  # Non-zero blocks (num_blocks, block_h, block_w)
        "indices": bsr_matrix.indices,  # Column indices of blocks
        "indptr": bsr_matrix.indptr,  # Row pointer (CSR-like structure)
        "shape": bsr_matrix.shape,  # Original matrix shape
        "blocksize": bsr_matrix.blocksize,  # (block_h, block_w)
    }


def build_bsr_from_dense(weight: np.ndarray, block_h: int, block_w: int, threshold: float = 1e-10) -> Dict:
    """
    Build BSR format directly from dense weight matrix.
    Skips padded zero blocks automatically.

    Args:
        weight: Dense weight matrix
        block_h: Block height
        block_w: Block width
        threshold: Blocks with L2 norm below this are considered zero

    Returns:
        BSR format dictionary with metadata
    """
    height, width = weight.shape

    # Pad to block size
    pad_h = (block_h - height % block_h) % block_h
    pad_w = (block_w - width % block_w) % block_w
    if pad_h > 0 or pad_w > 0:
        weight = np.pad(weight, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)

    padded_h, padded_w = weight.shape
    num_block_rows = padded_h // block_h
    num_block_cols = padded_w // block_w

    # Extract all blocks and compute norms
    blocks_list = []
    col_indices_list = []
    row_ptr = [0]

    num_nonzero_blocks = 0

    for block_row in range(num_block_rows):
        row_blocks_start = num_nonzero_blocks

        for block_col in range(num_block_cols):
            # Extract block
            r_start = block_row * block_h
            r_end = r_start + block_h
            c_start = block_col * block_w
            c_end = c_start + block_w

            block = weight[r_start:r_end, c_start:c_end]

            # Check if block is non-zero
            block_norm = np.linalg.norm(block)
            if block_norm > threshold:
                blocks_list.append(block)
                col_indices_list.append(block_col)
                num_nonzero_blocks += 1

        row_ptr.append(num_nonzero_blocks)

    # Convert to arrays
    if len(blocks_list) > 0:
        data = np.stack(blocks_list, axis=0)  # (num_blocks, block_h, block_w)
    else:
        data = np.zeros((0, block_h, block_w), dtype=weight.dtype)

    indices = np.array(col_indices_list, dtype=np.int32)
    indptr = np.array(row_ptr, dtype=np.int32)

    density = num_nonzero_blocks / (num_block_rows * num_block_cols) if num_block_rows * num_block_cols > 0 else 0.0

    return {
        "data": data,
        "indices": indices,
        "indptr": indptr,
        "shape": (height, width),  # Original shape (before padding)
        "padded_shape": (padded_h, padded_w),
        "blocksize": (block_h, block_w),
        "num_blocks": num_nonzero_blocks,
        "num_block_rows": num_block_rows,
        "num_block_cols": num_block_cols,
        "density": density,
        "sparsity_pct": (1.0 - density) * 100.0,
    }


# ----------------------------
# Binary Export Functions
# ----------------------------
def save_bsr_binary(bsr_data: Dict, filepath: str):
    """
    Save BSR data blocks as binary file.
    Format: [block1_data, block2_data, ...]
    Each block is flattened in row-major order.
    """
    data = bsr_data["data"]  # (num_blocks, block_h, block_w)

    with open(filepath, "wb") as f:
        # Write all blocks
        for block in data:
            # Flatten block in row-major (C order) and write as float32
            block_flat = block.flatten("C").astype(np.float32)
            f.write(block_flat.tobytes())

    print(f"  Saved binary: {filepath} ({os.path.getsize(filepath)} bytes)")


def save_bsr_binary_int8(bsr_data: Dict, scales: np.ndarray, filepath: str):
    """
    Save BSR data blocks as INT8 binary file with per-channel quantization.
    """
    data = bsr_data["data"]  # (num_blocks, block_h, block_w)
    block_h, block_w = bsr_data["blocksize"]

    with open(filepath, "wb") as f:
        for block_idx, block in enumerate(data):
            # Determine which output channel this block belongs to
            # This depends on block row mapping
            row_idx = np.searchsorted(bsr_data["indptr"], block_idx + 1) - 1

            # Get scale for this channel (if available)
            if scales.ndim > 0:
                scale = scales[row_idx] if row_idx < len(scales) else scales[0]
            else:
                scale = float(scales)

            # Quantize block to INT8
            block_int8 = np.clip(np.rint(block / scale), -128, 127).astype(np.int8)

            # Write as INT8
            f.write(block_int8.tobytes())

    print(f"  Saved INT8 binary: {filepath} ({os.path.getsize(filepath)} bytes)")


def save_bsr_metadata(bsr_data: Dict, filepath: str, layer_name: str = ""):
    """
    Save BSR metadata as JSON for hardware scheduler.

    Metadata includes:
    - Row pointer (indptr): where each block-row starts in data array
    - Column indices: which block-column each non-zero block is in
    - Block dimensions and matrix shape
    - Sparsity statistics
    """
    metadata = {
        "layer_name": layer_name,
        "shape": bsr_data["shape"],
        "padded_shape": bsr_data.get("padded_shape", bsr_data["shape"]),
        "blocksize": bsr_data["blocksize"],
        "num_blocks": int(bsr_data["num_blocks"]),
        "num_block_rows": int(bsr_data["num_block_rows"]),
        "num_block_cols": int(bsr_data["num_block_cols"]),
        "density": float(bsr_data["density"]),
        "sparsity_pct": float((1.0 - bsr_data["density"]) * 100),
        # Critical for hardware scheduler
        "row_ptr": bsr_data["indptr"].tolist(),  # Length: num_block_rows + 1
        "col_idx": bsr_data["indices"].tolist(),  # Length: num_blocks
        # Hardware scheduling hints
        "tiles_per_row": [
            int(bsr_data["indptr"][i + 1] - bsr_data["indptr"][i]) for i in range(bsr_data["num_block_rows"])
        ],
        "max_tiles_per_row": int(np.max(np.diff(bsr_data["indptr"]))),
        "avg_tiles_per_row": (
            float(bsr_data["num_blocks"] / bsr_data["num_block_rows"]) if bsr_data["num_block_rows"] > 0 else 0.0
        ),
    }

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata: {filepath}")


# ----------------------------
# Layer Export Functions
# ----------------------------
def export_layer_bsr(
    name: str,
    weight: np.ndarray,
    block_h: int,
    block_w: int,
    output_dir: str,
    quantize: bool = False,
    scales: Optional[np.ndarray] = None,
):
    """
    Export a single layer's weights in BSR format.

    Args:
        name: Layer name
        weight: Weight matrix
        block_h, block_w: Block dimensions
        output_dir: Output directory
        quantize: Whether to export INT8 quantized version
        scales: Per-channel quantization scales (if quantize=True)
    """
    print(f"\nExporting {name}:")
    print(f"  Weight shape: {weight.shape}")
    print(f"  Block size: {block_h}×{block_w}")

    # Build BSR format
    bsr_data = build_bsr_from_dense(weight, block_h, block_w)

    print(
        f"  Non-zero blocks: {bsr_data['num_blocks']}/{bsr_data['num_block_rows']*bsr_data['num_block_cols']} "
        f"({bsr_data['density']*100:.1f}% dense, {bsr_data['sparsity_pct']:.1f}% sparse)"
    )

    # Create output directory for this layer
    layer_dir = os.path.join(output_dir, name.replace(".", "_"))
    os.makedirs(layer_dir, exist_ok=True)

    # Save FP32 binary
    bsr_file = os.path.join(layer_dir, "weights.bsr")
    save_bsr_binary(bsr_data, bsr_file)

    # Save INT8 binary if requested
    if quantize and scales is not None:
        bsr_int8_file = os.path.join(layer_dir, "weights_int8.bsr")
        save_bsr_binary_int8(bsr_data, scales, bsr_int8_file)

    # Save metadata
    meta_file = os.path.join(layer_dir, "weights.meta.json")
    save_bsr_metadata(bsr_data, meta_file, layer_name=name)

    # Save indices and row_ptr as separate binary files for easy hardware loading
    np.save(os.path.join(layer_dir, "col_idx.npy"), bsr_data["indices"])
    np.save(os.path.join(layer_dir, "row_ptr.npy"), bsr_data["indptr"])

    return bsr_data


# ----------------------------
# Full Model Export
# ----------------------------
def export_model_bsr(model: nn.Module, output_dir: str, quantize: bool = False):
    """
    Export all layers of a sparse model in BSR format.

    Args:
        model: PyTorch model with sparse weights
        output_dir: Output directory
        quantize: Whether to also export INT8 quantized versions
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("BSR Export for ACCEL-BSR Hardware")
    print("=" * 70)

    # Summary statistics
    total_params = 0
    total_nonzero_blocks = 0
    total_blocks = 0

    layer_stats = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            weight = module.weight.data.cpu().numpy().astype(np.float32)

            # Get block configuration for this layer
            (block_h, block_w), _ = layer_block_cfg(name, module)

            # Handle Conv layers (flatten to 2D)
            original_shape = weight.shape
            if len(weight.shape) == 4:
                weight = weight.reshape(weight.shape[0], -1)

            # Export layer
            bsr_data = export_layer_bsr(
                name=name,
                weight=weight,
                block_h=block_h,
                block_w=block_w,
                output_dir=output_dir,
                quantize=quantize,
                scales=quantizer.scale_matrix if hasattr(quantizer, "scale_matrix") and quantize else None,  # Load from quantization if available
            )

            # Update statistics
            layer_params = np.prod(original_shape)
            total_params += layer_params
            total_nonzero_blocks += bsr_data["num_blocks"]
            total_blocks += bsr_data["num_block_rows"] * bsr_data["num_block_cols"]

            layer_stats.append(
                {
                    "name": name,
                    "original_shape": original_shape,
                    "blocksize": (block_h, block_w),
                    "num_blocks": bsr_data["num_blocks"],
                    "total_blocks": bsr_data["num_block_rows"] * bsr_data["num_block_cols"],
                    "density": bsr_data["density"],
                    "sparsity_pct": bsr_data["sparsity_pct"],
                }
            )

    # Save summary
    summary = {
        "model": "ACCEL-BSR MNIST",
        "total_parameters": int(total_params),
        "total_blocks": int(total_blocks),
        "nonzero_blocks": int(total_nonzero_blocks),
        "overall_block_density": float(total_nonzero_blocks / total_blocks) if total_blocks > 0 else 0.0,
        "overall_block_sparsity_pct": (
            float((1.0 - total_nonzero_blocks / total_blocks) * 100) if total_blocks > 0 else 0.0
        ),
        "layers": layer_stats,
    }

    summary_file = os.path.join(output_dir, "model_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("Export Summary")
    print("=" * 70)
    print(f"Total parameters: {total_params:,}")
    print(f"Total blocks: {total_blocks:,}")
    print(f"Non-zero blocks: {total_nonzero_blocks:,}")
    print(f"Block sparsity: {summary['overall_block_sparsity_pct']:.1f}%")
    print(f"\nOutput directory: {output_dir}")
    print("  - Each layer has:")
    print("    * weights.bsr (binary FP32 blocks)")
    print("    * weights.meta.json (row_ptr, col_idx, metadata)")
    print("    * col_idx.npy, row_ptr.npy (NumPy arrays)")
    print(f"  - model_summary.json (overall statistics)")

    return summary


# ----------------------------
# Load Sparse Model
# ----------------------------
def load_sparse_model_from_npz(npz_path: str, model: nn.Module) -> nn.Module:
    """
    Load sparse model weights from NPZ file (output of blocksparse_train.py).
    """
    if not os.path.exists(npz_path):
        print(f"Warning: Sparse model not found at {npz_path}")
        print("Using dense model weights (will still work, just not sparse)")
        return model

    print(f"Loading sparse weights from {npz_path}...")
    sparse_data = np.load(npz_path, allow_pickle=True)

    # Load weights back into model
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Reconstruct weight matrix from BSR data
            key_data = f"{name}_data"
            key_indices = f"{name}_indices"
            key_indptr = f"{name}_indptr"
            key_shape = f"{name}_shape"
            key_blocksize = f"{name}_blocksize"

            if key_data in sparse_data:
                bsr_blocks = sparse_data[key_data]
                bsr_indices = sparse_data[key_indices]
                bsr_indptr = sparse_data[key_indptr]
                original_shape = tuple(sparse_data[key_shape])
                blocksize = tuple(sparse_data[key_blocksize])

                # Determine padded shape for BSR reconstruction
                if len(original_shape) == 2:
                    height, width = original_shape
                else:  # Conv layer
                    height = original_shape[0]
                    width = np.prod(original_shape[1:])

                # Calculate padded dimensions
                block_h, block_w = blocksize
                pad_h = (block_h - height % block_h) % block_h
                pad_w = (block_w - width % block_w) % block_w
                padded_h = height + pad_h
                padded_w = width + pad_w

                # Reconstruct from BSR with padded shape
                bsr_matrix = sp.bsr_matrix((bsr_blocks, bsr_indices, bsr_indptr), shape=(padded_h, padded_w))
                weight_dense = bsr_matrix.toarray()

                # Remove padding
                weight_dense = weight_dense[:height, :width]

                # Reshape if conv layer
                if len(original_shape) == 4:
                    weight_dense = weight_dense.reshape(original_shape)

                # Load into model
                module.weight.data = torch.from_numpy(weight_dense).float()
                print(f"  Loaded {name}: {original_shape}, {len(bsr_blocks)} non-zero blocks")

    return model


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    # Create model
    model = Net().eval()

    # Try to load sparse weights
    if os.path.exists(SPARSE_MODEL_PATH):
        model = load_sparse_model_from_npz(SPARSE_MODEL_PATH, model)
    else:
        print(f"Warning: {SPARSE_MODEL_PATH} not found")
        print("Run blocksparse_train.py first to generate sparse model")
        print("Proceeding with dense model for demonstration...")

    # Export to BSR format
    summary = export_model_bsr(
        model=model, output_dir=OUTPUT_DIR, quantize=False  # Set to True when quantization is integrated
    )

    print("\n" + "=" * 70)
    print("BSR export complete! Ready for hardware integration.")
    print("=" * 70)
