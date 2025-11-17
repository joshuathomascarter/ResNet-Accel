# python/quantize.py
import os, json, numpy as np, torch
import torch.nn as nn
from torchvision import datasets, transforms

# ----------------------------
# 0) Paths & Tiling Parameters
# ----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # repo root
CKPT_PATH = os.path.join(ROOT, "data", "checkpoints", "mnist_fp32.pt")
GOLDEN_INPUTS_PATH = os.path.join(ROOT, "python", "golden", "mnist_inputs.npy")
OUT_DIR = os.path.join(ROOT, "data", "int8")

# RS tiling (small starter config)
Tm, Tn, Tk = 2, 2, 64  # A: (Tm×Tk), B: (Tk×Tn)
NUM_GOLDEN = max(32, Tm)  # how many test images to sample for activations

# ----------------------------
# 1) Load FP32 model checkpoint
# ----------------------------
ckpt = torch.load(CKPT_PATH, map_location="cpu")
state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt


# ----------------------------
# 2) Define the network (must match training)
# ----------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # after pooling, output is 64x12x12
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.c2, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
        x = torch.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net().eval()
model.load_state_dict(state_dict)

# ----------------------------
# 3) Get golden inputs (prefer saved file, else MNIST)
# ----------------------------
if os.path.isfile(GOLDEN_INPUTS_PATH):
    imgs = np.load(GOLDEN_INPUTS_PATH)[:NUM_GOLDEN]  # (N, 28, 28), uint8 or float
    x = torch.from_numpy(imgs).unsqueeze(1).float() / 255.0  # (N,1,28,28)
else:
    tfm = transforms.Compose([transforms.ToTensor()])  # Normalize next line
    test_set = datasets.MNIST(root=os.path.join(ROOT, "data"), train=False, download=True, transform=tfm)
    loader = torch.utils.data.DataLoader(test_set, batch_size=NUM_GOLDEN, shuffle=False, num_workers=0)
    x, _ = next(iter(loader))  # (N,1,28,28)
# Normalize exactly like training
x = (x - 0.1307) / 0.3081

# ----------------------------
# 4) Compute real FP32 activations up to flatten
# ----------------------------
with torch.no_grad():
    a = torch.relu(model.conv1(x))
    a = torch.relu(model.conv2(a))
    a = torch.nn.functional.max_pool2d(a, 2)  # -> (N, 64, 12, 12)
    a = torch.flatten(a, 1)  # -> (N, 64*12*12) = (N, 9216)

# Sanity: ensure we can slice requested tile
assert a.shape[0] >= Tm, f"Need at least Tm={Tm} samples, got {a.shape[0]}"
assert a.shape[1] >= Tk, f"Activation features {a.shape[1]} < Tk={Tk}"

# Take activation tile A in FP32: (Tm × Tk)
A_fp32 = a[:Tm, :Tk].cpu().numpy().astype(np.float32)

# ----------------------------
# 5) Build weight tile B in FP32 as (Tk × Tn)
# ----------------------------
# FC1 weight W has shape (N_out, K_in) = (128, 9216). We need B = W^T[:Tk, :Tn] -> (Tk, Tn).
W = state_dict["fc1.weight"].cpu().numpy().astype(np.float32)  # (128, 9216)
W_T = W.T  # (9216, 128) = (K, N)
assert W_T.shape[0] >= Tk and W_T.shape[1] >= Tn, "FC1 weight too small for requested (Tk,Tn)"
B_fp32 = W_T[:Tk, :Tn]  # (Tk, Tn)


# ----------------------------
# 6) Symmetric per-tensor INT8 quantization
# ----------------------------
def quantize_symmetric_int8(x: np.ndarray):
    # scale maps max|x| -> 127; guard for all-zeros
    maxabs = float(np.max(np.abs(x)))
    scale = max(maxabs / 127.0, 1e-12)
    q = np.rint(x / scale)  # banker's rounding
    q = np.clip(q, -128, 127).astype(np.int8)
    return q, scale


A_int8, Sx = quantize_symmetric_int8(A_fp32)  # A: per-tensor
B_int8, Sw = quantize_symmetric_int8(B_fp32)  # B: per-layer (here same tensor)

# Quick dequant error stats (for log/debug)
A_deq = A_int8.astype(np.float32) * Sx
B_deq = B_int8.astype(np.float32) * Sw
A_err_max = float(np.max(np.abs(A_fp32 - A_deq)))
B_err_max = float(np.max(np.abs(B_fp32 - B_deq)))
A_err_mae = float(np.mean(np.abs(A_fp32 - A_deq)))
B_err_mae = float(np.mean(np.abs(B_fp32 - B_deq)))

# ----------------------------
# 7) Save tiles + scales
# ----------------------------
os.makedirs(OUT_DIR, exist_ok=True)
np.save(os.path.join(OUT_DIR, "A.npy"), A_int8)  # (Tm, Tk)
np.save(os.path.join(OUT_DIR, "B.npy"), B_int8)  # (Tk, Tn)
with open(os.path.join(OUT_DIR, "scales.json"), "w") as f:
    json.dump(
        {
            "Tm": Tm,
            "Tn": Tn,
            "Tk": Tk,
            "Sx": float(Sx),
            "Sw": float(Sw),
            "A_err_max": A_err_max,
            "A_err_mae": A_err_mae,
            "B_err_max": B_err_max,
            "B_err_mae": B_err_mae,
        },
        f,
        indent=2,
    )

print(f"[OK] Exported INT8 tiles to {OUT_DIR}")
print(
    f"  A.npy shape: {A_int8.shape}  (expected {Tm}×{Tk}),  Sx={Sx:.6g},  max|err|={A_err_max:.6g}, MAE={A_err_mae:.6g}"
)
print(
    f"  B.npy shape: {B_int8.shape}  (expected {Tk}×{Tn}),  Sw={Sw:.6g},  max|err|={B_err_max:.6g}, MAE={B_err_mae:.6g}"
)
