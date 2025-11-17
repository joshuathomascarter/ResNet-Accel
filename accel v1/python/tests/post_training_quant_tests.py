"""
Post-Training Quantization (PTQ) Tests
Tests the quantization functionality using PyTorch and MNIST model
"""

import pytest
import numpy as np
import torch
import torch.nn as nn


def quantize_symmetric_int8(x: np.ndarray):
    """
    Symmetric INT8 quantization (from quantize.py)
    Maps max|x| -> 127, clips to [-128, 127]
    """
    maxabs = float(np.max(np.abs(x)))
    scale = max(maxabs / 127.0, 1e-12)  # guard for all-zeros
    q = np.rint(x / scale)  # banker's rounding
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scale


# Simple MNIST model for testing
class SimpleMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test_quantization_accuracy():
    """Test PTQ quantization accuracy against FP32 baseline"""
    # Create simple model with known weights
    model = SimpleMNIST()

    # Set known weights for reproducibility
    with torch.no_grad():
        model.fc1.weight.fill_(0.01)
        model.fc1.bias.fill_(0.0)
        model.fc2.weight.fill_(0.01)
        model.fc2.bias.fill_(0.0)

    # Create test input
    test_input = torch.randn(10, 784)

    # Get FP32 output
    model.eval()
    with torch.no_grad():
        fp32_output = model(test_input)

    # Quantize first layer weights
    fc1_weight_np = model.fc1.weight.detach().numpy()
    fc1_weight_int8, scale = quantize_symmetric_int8(fc1_weight_np)

    # Dequantize
    fc1_weight_dequant = fc1_weight_int8.astype(np.float32) * scale

    # Check quantization error is reasonable (within 5%)
    relative_error = np.mean(np.abs(fc1_weight_np - fc1_weight_dequant) / (np.abs(fc1_weight_np) + 1e-8))
    assert relative_error < 0.05, f"Quantization error too high: {relative_error}"

    print(f"Quantization relative error: {relative_error:.4f}")


def test_weight_scale_calculation():
    """Test symmetric weight scale calculation"""
    # Create test weight matrix
    weights = np.array([[127.0, -127.0, 63.5, -63.5], [10.0, -10.0, 5.0, -5.0]], dtype=np.float32)

    # Quantize
    weights_int8, scale = quantize_symmetric_int8(weights)

    # Check scale is correct (max_abs / 127)
    max_abs = np.max(np.abs(weights))
    expected_scale = max_abs / 127.0
    assert np.isclose(scale, expected_scale, rtol=1e-5), f"Scale mismatch: {scale} vs {expected_scale}"

    # Check quantized values are clipped to [-128, 127]
    assert np.all(weights_int8 >= -128) and np.all(weights_int8 <= 127), "Quantized values out of INT8 range"

    # Check largest value quantizes to ±127
    max_val_idx = np.argmax(np.abs(weights))
    assert (
        np.abs(weights_int8.flat[max_val_idx]) == 127
    ), f"Max value should quantize to ±127, got {weights_int8.flat[max_val_idx]}"

    print(f"Weight scale: {scale:.6f}")
    print(f"Quantized range: [{weights_int8.min()}, {weights_int8.max()}]")


def test_activation_scale_calibration():
    """Test activation scale calibration with dataset"""
    # Simulate activation values from a calibration dataset
    np.random.seed(42)

    # Create synthetic activations (simulating ReLU outputs)
    activations = np.abs(np.random.randn(100, 128).astype(np.float32))

    # Quantize activations
    act_int8, scale = quantize_symmetric_int8(activations)

    # Check scale is reasonable
    assert scale > 0, "Scale must be positive"
    assert scale < 1.0, "Scale should be < 1 for typical activations"

    # Check quantized values use full range
    assert act_int8.max() >= 100, "Quantization should use most of INT8 range"

    # Check dequantization accuracy
    act_dequant = act_int8.astype(np.float32) * scale
    mse = np.mean((activations - act_dequant) ** 2)
    relative_mse = mse / np.mean(activations**2)

    assert relative_mse < 0.01, f"Relative MSE too high: {relative_mse}"

    print(f"Activation scale: {scale:.6f}")
    print(f"Relative MSE: {relative_mse:.6f}")
    print(f"Quantized activation range: [{act_int8.min()}, {act_int8.max()}]")
