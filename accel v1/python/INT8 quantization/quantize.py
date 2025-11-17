"""
INT8 Per-Channel Quantization for ACCEL-BSR
Extends basic quantization to support:
- Per-channel quantization for better accuracy
- Block-sparse weight handling
- Full model quantization (not just tiles)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from typing import Dict, Tuple, Optional

# ----------------------------
# Configuration
# ----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_PATH = os.path.join(ROOT, "data", "checkpoints", "mnist_fp32.pt")
SPARSE_CKPT_PATH = os.path.join(ROOT, "python", "training", "mnist_sparse_90pct.npz")
GOLDEN_INPUTS_PATH = os.path.join(ROOT, "python", "golden", "mnist_inputs.npy")
OUT_DIR = os.path.join(ROOT, "data", "int8")

# Tiling parameters for hardware
Tm, Tn, Tk = 2, 2, 64
NUM_GOLDEN = max(32, Tm)


# ----------------------------
# Model Definition
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


# ----------------------------
# Quantization Functions
# ----------------------------
def quantize_symmetric_per_tensor(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Symmetric per-tensor INT8 quantization.
    Maps max|x| -> 127

    Returns:
        q: INT8 quantized tensor
        scale: Quantization scale factor
    """
    maxabs = float(np.max(np.abs(x)))
    scale = max(maxabs / 127.0, 1e-12)
    q = np.rint(x / scale)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scale


def quantize_symmetric_per_channel(x: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric per-channel INT8 quantization.
    Each channel (along specified axis) gets its own scale.

    Args:
        x: Input array (e.g., weights of shape (out_channels, in_channels, ...))
        axis: Channel axis (typically 0 for output channels)

    Returns:
        q: INT8 quantized tensor
        scales: Per-channel scale factors (shape matching channel dimension)
    """
    # Compute max abs per channel
    axes_to_reduce = tuple(i for i in range(len(x.shape)) if i != axis)
    maxabs = np.max(np.abs(x), axis=axes_to_reduce, keepdims=True)

    # Guard against all-zero channels
    scales = np.maximum(maxabs / 127.0, 1e-12)

    # Quantize
    q = np.rint(x / scales)
    q = np.clip(q, -128, 127).astype(np.int8)

    # Flatten scales for storage
    scales_flat = np.squeeze(scales, axis=axes_to_reduce).astype(np.float32)

    return q, scales_flat


def quantize_asymmetric_per_channel(x: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Asymmetric per-channel UINT8 quantization (for activations).
    Each channel gets its own scale and zero-point.

    Args:
        x: Input array
        axis: Channel axis

    Returns:
        q: UINT8 quantized tensor
        scales: Per-channel scale factors
        zero_points: Per-channel zero points
    """
    axes_to_reduce = tuple(i for i in range(len(x.shape)) if i != axis)

    # Find min and max per channel
    x_min = np.min(x, axis=axes_to_reduce, keepdims=True)
    x_max = np.max(x, axis=axes_to_reduce, keepdims=True)

    # Compute scale and zero-point
    scales = (x_max - x_min) / 255.0
    scales = np.maximum(scales, 1e-12)  # Guard against zero range
    zero_points = np.rint(-x_min / scales)
    # Keep zero_points as signed int32 (can be negative for optimal range usage)

    # Quantize
    q = np.rint(x / scales + zero_points)
    q = np.clip(q, 0, 255).astype(np.uint8)

    # Flatten for storage
    scales_flat = np.squeeze(scales, axis=axes_to_reduce).astype(np.float32)
    zero_points_flat = np.squeeze(zero_points, axis=axes_to_reduce).astype(np.int32)

    return q, scales_flat, zero_points_flat


def compute_quantization_error(x_fp32: np.ndarray, x_int8: np.ndarray, scale: np.ndarray) -> Dict[str, float]:
    """Compute quantization error statistics"""
    if isinstance(scale, np.ndarray) and scale.ndim > 0:
        # Per-channel: broadcast scales
        shape = [1] * len(x_fp32.shape)
        shape[0] = len(scale)
        scale_broadcast = scale.reshape(shape)
        x_deq = x_int8.astype(np.float32) * scale_broadcast
    else:
        # Per-tensor
        x_deq = x_int8.astype(np.float32) * scale

    error = np.abs(x_fp32 - x_deq)
    return {
        "max_error": float(np.max(error)),
        "mean_error": float(np.mean(error)),
        "mse": float(np.mean(error**2)),
        "snr_db": float(20 * np.log10(np.std(x_fp32) / (np.std(error) + 1e-12))),
    }


# ----------------------------
# Full Model Quantization
# ----------------------------
def quantize_model_per_channel(model: nn.Module, use_sparse: bool = False) -> Dict:
    """
    Quantize all model weights using per-channel INT8.

    Args:
        model: PyTorch model
        use_sparse: Whether to use sparse weights from BSR export

    Returns:
        Dictionary with quantized weights, scales, and metadata
    """
    quantized_model = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            weight = module.weight.data.cpu().numpy().astype(np.float32)

            # Determine quantization axis (output channels)
            axis = 0

            # Per-channel quantization
            weight_int8, weight_scales = quantize_symmetric_per_channel(weight, axis=axis)

            # Compute error statistics
            error_stats = compute_quantization_error(weight, weight_int8, weight_scales)

            # Store quantized weights and metadata
            quantized_model[f"{name}.weight"] = {
                "data": weight_int8,
                "scales": weight_scales,
                "shape": weight.shape,
                "axis": axis,
                "error": error_stats,
            }

            # Quantize bias if present (per-tensor is fine for bias)
            if module.bias is not None:
                bias = module.bias.data.cpu().numpy().astype(np.float32)
                bias_int8, bias_scale = quantize_symmetric_per_tensor(bias)
                bias_error = compute_quantization_error(bias, bias_int8, bias_scale)

                quantized_model[f"{name}.bias"] = {
                    "data": bias_int8,
                    "scale": bias_scale,
                    "shape": bias.shape,
                    "error": bias_error,
                }

            print(f"Quantized {name}:")
            print(f"  Weight: {weight.shape} -> INT8 with {len(weight_scales)} per-channel scales")
            print(f"  Error: max={error_stats['max_error']:.6f}, SNR={error_stats['snr_db']:.2f} dB")

    return quantized_model


def quantize_activations_from_golden(model: nn.Module, num_samples: int = 32) -> Dict:
    """
    Run model on golden inputs and quantize activations per-channel.

    Returns:
        Dictionary with quantized activation tiles and scales
    """
    # Load or generate golden inputs
    if os.path.isfile(GOLDEN_INPUTS_PATH):
        imgs = np.load(GOLDEN_INPUTS_PATH)[:num_samples]
        x = torch.from_numpy(imgs).unsqueeze(1).float() / 255.0
    else:
        tfm = transforms.Compose([transforms.ToTensor()])
        test_set = datasets.MNIST(root=os.path.join(ROOT, "data"), train=False, download=True, transform=tfm)
        loader = torch.utils.data.DataLoader(test_set, batch_size=num_samples, shuffle=False, num_workers=0)
        x, _ = next(iter(loader))

    # Normalize like training
    x = (x - 0.1307) / 0.3081

    # Run through model to get activations
    activations = {}
    with torch.no_grad():
        a = torch.relu(model.conv1(x))
        activations["conv1_out"] = a.cpu().numpy().astype(np.float32)

        a = torch.relu(model.conv2(a))
        activations["conv2_out"] = a.cpu().numpy().astype(np.float32)

        a = torch.nn.functional.max_pool2d(a, 2)
        a_flat = torch.flatten(a, 1)
        activations["fc1_in"] = a_flat.cpu().numpy().astype(np.float32)

        a = torch.relu(model.fc1(a_flat))
        activations["fc1_out"] = a.cpu().numpy().astype(np.float32)

    # Quantize activations (asymmetric for better range coverage)
    quantized_acts = {}
    for name, act in activations.items():
        # Use per-tensor for activations (simpler for hardware)
        act_int8, act_scale = quantize_symmetric_per_tensor(act)
        error = compute_quantization_error(act, act_int8, act_scale)

        quantized_acts[name] = {"data": act_int8, "scale": act_scale, "shape": act.shape, "error": error}

        print(f"Quantized {name}: shape={act.shape}, scale={act_scale:.6f}, SNR={error['snr_db']:.2f} dB")

    return quantized_acts


# ----------------------------
# Save Functions
# ----------------------------
def save_quantized_model(quantized_model: Dict, output_dir: str):
    """Save quantized model to disk"""
    os.makedirs(output_dir, exist_ok=True)

    # Save each layer's quantized weights
    for param_name, param_data in quantized_model.items():
        layer_name = param_name.replace(".", "_")

        # Save INT8 data
        np.save(os.path.join(output_dir, f"{layer_name}_int8.npy"), param_data["data"])

        # Save scales
        if "scales" in param_data:
            np.save(os.path.join(output_dir, f"{layer_name}_scales.npy"), param_data["scales"])
        elif "scale" in param_data:
            with open(os.path.join(output_dir, f"{layer_name}_scale.json"), "w") as f:
                json.dump({"scale": float(param_data["scale"])}, f, indent=2)

    # Save metadata
    metadata = {}
    for param_name, param_data in quantized_model.items():
        metadata[param_name] = {
            "shape": param_data["shape"],
            "quantization": "per_channel" if "scales" in param_data else "per_tensor",
            "error": param_data["error"],
        }

    with open(os.path.join(output_dir, "quantization_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[OK] Saved quantized model to {output_dir}")


def save_hardware_tiles(quantized_acts: Dict, quantized_model: Dict, output_dir: str):
    """
    Save specific tiles for hardware testing (A and B matrices for GEMM).
    """
    # Extract FC1 input activations as A matrix
    fc1_in = quantized_acts["fc1_in"]
    A_fp32 = fc1_in["data"][:Tm, :Tk].astype(np.float32) * fc1_in["scale"]
    A_int8 = A_fp32 / fc1_in["scale"]
    A_int8 = np.clip(np.rint(A_int8), -128, 127).astype(np.int8)

    # Extract FC1 weights as B matrix (transposed)
    fc1_weight = quantized_model["fc1.weight"]
    W_int8 = fc1_weight["data"]  # (128, 9216)
    W_scales = fc1_weight["scales"]  # (128,)

    # Transpose and extract tile
    W_T = W_int8.T  # (9216, 128)
    B_int8 = W_T[:Tk, :Tn]
    B_scales = W_scales[:Tn]

    # Save tiles
    tile_dir = os.path.join(output_dir, "tiles")
    os.makedirs(tile_dir, exist_ok=True)

    np.save(os.path.join(tile_dir, "A.npy"), A_int8)
    np.save(os.path.join(tile_dir, "B.npy"), B_int8)

    with open(os.path.join(tile_dir, "scales.json"), "w") as f:
        json.dump(
            {
                "Tm": Tm,
                "Tn": Tn,
                "Tk": Tk,
                "Sx": float(fc1_in["scale"]),
                "Sw": B_scales.tolist(),  # Per-channel scales for B
                "quantization": "per_channel",
            },
            f,
            indent=2,
        )

    print(f"\n[OK] Saved hardware tiles to {tile_dir}")
    print(f"  A.npy: {A_int8.shape} (Tm×Tk)")
    print(f"  B.npy: {B_int8.shape} (Tk×Tn)")
    print(f"  B uses per-channel scales: {len(B_scales)} channels")


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("ACCEL-BSR INT8 Per-Channel Quantization")
    print("=" * 60)

    # Load FP32 model
    print("\n1. Loading FP32 model...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    model = Net().eval()
    model.load_state_dict(state_dict)
    print(f"   Loaded from {CKPT_PATH}")

    # Quantize full model (per-channel)
    print("\n2. Quantizing model weights (per-channel INT8)...")
    quantized_model = quantize_model_per_channel(model)

    # Quantize activations from golden inputs
    print("\n3. Quantizing activations from golden inputs...")
    quantized_acts = quantize_activations_from_golden(model, num_samples=NUM_GOLDEN)

    # Save quantized model
    print("\n4. Saving quantized model...")
    save_quantized_model(quantized_model, OUT_DIR)

    # Save hardware test tiles
    print("\n5. Saving hardware test tiles...")
    save_hardware_tiles(quantized_acts, quantized_model, OUT_DIR)

    print("\n" + "=" * 60)
    print("Quantization complete!")
    print("=" * 60)
    print(f"\nOutput directory: {OUT_DIR}")
    print("  - Full model: *_int8.npy, *_scales.npy")
    print("  - Hardware tiles: tiles/A.npy, tiles/B.npy")
    print("  - Metadata: quantization_metadata.json")
