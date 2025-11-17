# Core PyTorch imports
import torch  # Main tensor operations and CUDA support
import torch.nn as nn  # Neural network layers and modules
import torch.optim as optim  # Optimizers (Adam, SGD) for training
import torch.nn.functional as F  # Activation functions (relu, etc.)

# Data loading and preprocessing
from torchvision import datasets, transforms  # MNIST dataset and image transforms
from torch.utils.data import DataLoader  # Batch loading for training

# System and file operations
import os  # Directory creation and file paths
import random  # Random seed for reproducibility
import numpy as np  # Array operations and BSR format export
import matplotlib.pyplot as plt  # Saving misclassified images (optional)

# Type hints for clean code
from typing import Tuple, Dict, List, Optional  # Function signatures

# Sparse matrix operations
import scipy.sparse as sp  # BSR (Block Sparse Row) format export

# Math operations
import math  # Block size calculations and rounding

# Progress tracking (optional)
from tqdm import tqdm  # Progress bars for long training loops


# Define the same CNN model architecture from original training
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # after pooling, output is 64x12x12
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def layer_block_cfg(name, module):
    """Return block size and minimum keep percentage for each layer"""
    if isinstance(module, nn.Conv2d):
        return (4, 4), 0.30  # 4x4 blocks, keep >= 30%
    else:  # Linear
        return (8, 8), 0.05  # 8x8 blocks, keep >= 5%


def load_dense_model() -> Tuple[nn.Module, torch.device, float, Dict[str, torch.Tensor]]:
    """Load the pre-trained dense MNIST model from checkpoint"""

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = "../../data/checkpoints/mnist_fp32.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Dense model checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model and load weights
    model = Net()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Get original accuracy for comparison
    original_acc = checkpoint.get("best_acc", 0.0)
    print(f"Loaded dense model with original accuracy: {original_acc:.2f}%")

    # Create persistent masks for each layer
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            masks[name] = torch.ones_like(module.weight, dtype=torch.bool, device=device)

    return model, device, original_acc, masks


def compute_block_norms(
    weight_tensor: torch.Tensor, block_h: int, block_w: int
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, ...]]:
    """
    Compute L2 norms of blocks in a weight tensor.

    Args:
        weight_tensor: Input weight tensor
        block_h: Block height
        block_w: Block width

    Returns:
        Block norms, (num_blocks_h, num_blocks_w), original_shape
    """
    # Get original shape
    original_shape = weight_tensor.shape

    # Handle different layer types
    if len(original_shape) == 2:  # Linear layer
        height, width = original_shape
    elif len(original_shape) == 4:  # Conv layer
        weight_tensor = weight_tensor.view(original_shape[0], -1)
        height, width = weight_tensor.shape
    else:
        raise ValueError(f"Unsupported weight tensor shape: {original_shape}")

    # Pad tensor to be divisible by block size
    pad_height = (block_h - height % block_h) % block_h
    pad_width = (block_w - width % block_w) % block_w

    if pad_height > 0 or pad_width > 0:
        weight_tensor = F.pad(weight_tensor, (0, pad_width, 0, pad_height), value=0.0)
        height, width = weight_tensor.shape

    # Number of blocks
    num_blocks_h = height // block_h
    num_blocks_w = width // block_w

    # Reshape into blocks
    blocks = weight_tensor.view(num_blocks_h, block_h, num_blocks_w, block_w)
    blocks = blocks.permute(0, 2, 1, 3)  # (num_blocks_h, num_blocks_w, block_h, block_w)

    # Compute L2 norm for each block
    block_norms = torch.norm(blocks, p=2, dim=(2, 3))

    return block_norms, (num_blocks_h, num_blocks_w), original_shape


def prune_blocks_global(model: nn.Module, masks: Dict[str, torch.Tensor], target_sparsity: float) -> int:
    """
    Global block pruning across all layers with per-layer floors.

    Args:
        model: Neural network model
        masks: Persistent boolean masks
        target_sparsity: Global target sparsity (0.0 to 1.0)

    Returns:
        Number of blocks pruned
    """
    total_blocks_pruned = 0

    # Collect all block norms with layer info
    all_norms = []
    block_info = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            (block_h, block_w), min_keep = layer_block_cfg(name, module)
            block_norms, (num_blocks_h, num_blocks_w), _ = compute_block_norms(module.weight.data, block_h, block_w)

            # Flatten norms and store with layer info
            flat_norms = block_norms.flatten()
            for i, norm_val in enumerate(flat_norms):
                all_norms.append(norm_val.item())
                block_info.append((name, i, num_blocks_h, num_blocks_w, block_h, block_w, min_keep))

    # Sort all blocks by norm (weakest first)
    sorted_indices = sorted(range(len(all_norms)), key=lambda i: all_norms[i])

    # Calculate how many to prune globally
    total_blocks = len(all_norms)
    blocks_to_prune = int(total_blocks * target_sparsity)

    # Apply per-layer floors
    layer_block_counts = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            (block_h, block_w), min_keep = layer_block_cfg(name, module)
            _, (num_blocks_h, num_blocks_w), _ = compute_block_norms(module.weight.data, block_h, block_w)
            layer_block_counts[name] = num_blocks_h * num_blocks_w

    # Find blocks to prune, respecting floors
    to_prune = []
    layer_pruned = {name: 0 for name in layer_block_counts}

    for idx in sorted_indices:
        name, block_idx, num_blocks_h, num_blocks_w, block_h, block_w, min_keep = block_info[idx]
        layer_total = layer_block_counts[name]
        layer_min_keep = int(layer_total * min_keep)
        layer_current_kept = layer_total - layer_pruned[name]

        # Skip if would violate floor
        if layer_current_kept <= layer_min_keep:
            continue

        to_prune.append((name, block_idx, num_blocks_h, num_blocks_w, block_h, block_w))
        layer_pruned[name] += 1

        if len(to_prune) >= blocks_to_prune:
            break

    # Apply pruning to masks
    for name, block_idx, num_blocks_h, num_blocks_w, block_h, block_w in to_prune:
        # Convert flat index to 2D coordinates
        block_row = block_idx // num_blocks_w
        block_col = block_idx % num_blocks_w

        # Zero the mask for this block
        start_row = block_row * block_h
        end_row = start_row + block_h
        start_col = block_col * block_w
        end_col = start_col + block_w

        # Handle mask shape
        if len(masks[name].shape) == 2:
            masks[name][start_row:end_row, start_col:end_col] = False
        elif len(masks[name].shape) == 4:
            # For conv layers
            mask_2d = masks[name].view(masks[name].shape[0], -1)
            mask_2d[start_row:end_row, start_col:end_col] = False
            masks[name] = mask_2d.view(masks[name].shape)

    # Apply masks to weights
    apply_masks(model, masks)

    total_blocks_pruned = len(to_prune)
    print(f"Global pruning: {total_blocks_pruned}/{total_blocks} blocks ({target_sparsity*100:.1f}% sparse)")

    # Print per-layer stats
    for name in layer_block_counts:
        layer_total = layer_block_counts[name]
        layer_pruned_count = layer_pruned[name]
        layer_sparsity = layer_pruned_count / layer_total * 100
        print(f"  {name}: {layer_total - layer_pruned_count}/{layer_total} kept ({layer_sparsity:.1f}% sparse)")

    return total_blocks_pruned


def apply_masks(model: nn.Module, masks: Dict[str, torch.Tensor]):
    """Apply persistent masks to model weights"""
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.weight.mul_(masks[name])


def validate_accuracy(model: nn.Module, device: torch.device) -> float:
    """
    Validate model accuracy on MNIST test set.

    Args:
        model: Neural network model to validate
        device: CUDA device

    Returns:
        Test accuracy as percentage
    """
    # Set up test data loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root="../../data", train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def progressive_sparsity(
    model: nn.Module, device: torch.device, masks: Dict[str, torch.Tensor], target_sparsity: float = 0.9
):
    """
    Progressively prune model from dense to target sparsity with fine-tuning between phases.

    Args:
        model: Neural network model to sparsify
        device: CUDA device
        masks: Persistent boolean masks
        target_sparsity: Final target sparsity (default 0.9 = 90%)
    """
    # Progressive sparsity schedule: 50% → 70% → 85% → 90%
    sparsity_schedule = [0.5, 0.7, 0.85, target_sparsity]

    print(f"Starting progressive sparsification to {target_sparsity*100:.1f}%...")

    for phase, sparsity in enumerate(sparsity_schedule):
        print(f"\n=== Phase {phase+1}: Targeting {sparsity*100:.1f}% sparsity ===")

        # Prune to current sparsity level
        blocks_pruned = prune_blocks_global(model, masks, sparsity)

        # Validate accuracy after pruning
        accuracy = validate_accuracy(model, device)
        print(f"Accuracy after pruning: {accuracy:.2f}%")

        # Fine-tune to recover accuracy (always, including final phase)
        fine_tune_epochs = 10 if sparsity >= 0.9 else 3
        fine_tune_lr = 5e-5 if sparsity >= 0.9 else 1e-4
        print(f"Fine-tuning ({fine_tune_epochs} epochs) ...")
        train_with_group_lasso(
            model, device, masks, epochs=fine_tune_epochs, lr=fine_tune_lr, current_sparsity=sparsity
        )

        # Validate again after fine-tuning
        accuracy_after = validate_accuracy(model, device)
        print(f"Accuracy after fine-tuning: {accuracy_after:.2f}%")

    print(f"\nProgressive sparsification complete! Final sparsity: {target_sparsity*100:.1f}%")


def train_with_group_lasso(
    model: nn.Module,
    device: torch.device,
    masks: Dict[str, torch.Tensor],
    epochs: int = 3,
    lr: float = 1e-4,
    current_sparsity: float = 0.0,
):
    """
    Fine-tune model with group lasso regularization to encourage block sparsity.

    Args:
        model: Neural network model to train
        device: CUDA device
        masks: Persistent boolean masks
        epochs: Number of training epochs
        lr: Learning rate
        current_sparsity: Current sparsity level for regularization scheduling
    """
    # Set up data loaders (same as original training)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root="../../data", train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    # Set up optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            # Standard cross-entropy loss
            ce_loss = criterion(output, target)

            # Group lasso regularization with scheduling
            reg_loss = 0.0
            reg_base = 5e-4
            phase_alpha = 1.0 if current_sparsity < 0.85 else 0.5
            reg_weight = reg_base * phase_alpha

            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # Compute block norms and add L2,1 penalty
                    (block_h, block_w), _ = layer_block_cfg(name, module)
                    block_norms, _, _ = compute_block_norms(module.weight, block_h, block_w)
                    reg_loss += torch.sum(block_norms)  # L1 norm of L2 block norms

            total_loss = ce_loss + reg_weight * reg_loss
            total_loss.backward()
            optimizer.step()

            # Apply masks to keep pruned blocks zero
            apply_masks(model, masks)

            running_loss += total_loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

        accuracy = 100.0 * correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


def export_to_bsr_format(model: nn.Module, save_path: str = "sparse_weights_bsr.npz"):
    """
    Export sparse model weights to BSR (Block Sparse Row) format for hardware.

    Args:
        model: Sparse neural network model
        save_path: Path to save BSR format weights
    """
    bsr_data = {}

    print("Exporting sparse weights to BSR format...")

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            weight = module.weight.data.cpu().numpy()

            # Get block size for this layer
            (block_h, block_w), _ = layer_block_cfg(name, module)

            # Handle different layer shapes
            if len(weight.shape) == 4:  # Conv layer
                # Flatten conv weights to 2D
                original_shape = weight.shape
                weight = weight.reshape(weight.shape[0], -1)
            else:
                original_shape = weight.shape

            # Pad weight matrix to be divisible by block size for BSR format
            height, width = weight.shape
            pad_height = (block_h - height % block_h) % block_h
            pad_width = (block_w - width % block_w) % block_w

            if pad_height > 0 or pad_width > 0:
                # Pad with zeros
                weight = np.pad(weight, ((0, pad_height), (0, pad_width)), mode="constant", constant_values=0.0)

            # Convert to sparse matrix and then to BSR format with layer-specific block size
            sparse_matrix = sp.csr_matrix(weight)
            bsr_matrix = sparse_matrix.tobsr(blocksize=(block_h, block_w))

            # Store BSR components
            bsr_data[f"{name}_data"] = bsr_matrix.data
            bsr_data[f"{name}_indices"] = bsr_matrix.indices
            bsr_data[f"{name}_indptr"] = bsr_matrix.indptr
            bsr_data[f"{name}_shape"] = original_shape
            bsr_data[f"{name}_blocksize"] = bsr_matrix.blocksize

            # Calculate sparsity statistics
            total_blocks = (weight.shape[0] // block_h) * (weight.shape[1] // block_w)
            nonzero_blocks = len(bsr_matrix.data)
            sparsity = (1 - nonzero_blocks / total_blocks) * 100

            print(f"  {name}: {nonzero_blocks}/{total_blocks} blocks ({sparsity:.1f}% sparse)")

    # Save to compressed numpy file
    np.savez_compressed(save_path, **bsr_data)
    print(f"BSR weights saved to {save_path}")
    return save_path


# Main execution function
if __name__ == "__main__":
    # Load dense model
    model, device, original_acc, masks = load_dense_model()
    print(f"Original model accuracy: {original_acc:.2f}%")

    # Apply progressive sparsification
    progressive_sparsity(model, device, masks, target_sparsity=0.9)

    # Final validation
    final_acc = validate_accuracy(model, device)
    accuracy_loss = original_acc - final_acc
    print(f"\nFinal Results:")
    print(f"Original accuracy: {original_acc:.2f}%")
    print(f"Sparse accuracy: {final_acc:.2f}%")
    print(f"Accuracy loss: {accuracy_loss:.2f}%")

    # Export to BSR format for hardware
    export_to_bsr_format(model, "mnist_sparse_90pct.npz")

    print("\nBlock-sparse training complete! 🎯")
