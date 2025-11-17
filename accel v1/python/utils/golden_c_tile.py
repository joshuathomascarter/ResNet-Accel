"""
golden_c_tile.py

Reference Python model for C_tile (int32) and test-vector generator.
Generates a small set of random and corner-case vectors and saves them to
`tb/integration/test_vectors/example_vectors.npz`.

Usage:
    python python/utils/golden_c_tile.py --m 8 --n 8 --k 8 --count 4

The produced NPZ contains arrays: A_list (count, M, K), B_list (count, K, N), C_list (count, M, N)
"""

import argparse
import numpy as np
import os


def compute_c_tile(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute C = A * B (matrix multiply) with int32 arithmetic exactly.

    A: (M, K) int32
    B: (K, N) int32
    returns C: (M, N) int32
    """
    # Ensure int32 input
    A32 = A.astype(np.int64)  # use wider accumulator then cast back to int32
    B32 = B.astype(np.int64)
    M, K = A32.shape
    K2, N = B32.shape
    assert K == K2, "Matrix dimensions do not match for multiplication"
    C = np.zeros((M, N), dtype=np.int64)
    for i in range(M):
        for j in range(N):
            s = 0
            for k in range(K):
                s += int(A32[i, k]) * int(B32[k, j])
            C[i, j] = s
    # Saturate or wrap? We'll keep full int64 and then cast to int32 to match hardware two's complement wrap.
    C32 = (C & 0xFFFFFFFF).astype(np.int32)
    return C32


def gen_vectors(M: int, N: int, K: int, count: int, out_dir: str):
    """Generate test vectors and save them as .npz and .hex files."""
    rng = np.random.default_rng(12345)
    A_list = np.zeros((count, M, K), dtype=np.int32)
    B_list = np.zeros((count, K, N), dtype=np.int32)
    C_list = np.zeros((count, M, N), dtype=np.int32)

    for t in range(count):
        if t == 0:
            # simple identity-ish / deterministic small
            A = rng.integers(-4, 5, size=(M, K), dtype=np.int32)
            B = rng.integers(-4, 5, size=(K, N), dtype=np.int32)
        elif t == 1:
            # corner: large values to test wrap
            A = rng.integers(-(2**15), 2**15 - 1, size=(M, K), dtype=np.int32)
            B = rng.integers(-(2**15), 2**15 - 1, size=(K, N), dtype=np.int32)
        else:
            A = rng.integers(-128, 127, size=(M, K), dtype=np.int32)
            B = rng.integers(-128, 127, size=(K, N), dtype=np.int32)
        C = compute_c_tile(A, B)
        A_list[t] = A
        B_list[t] = B
        C_list[t] = C

    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "example_vectors.npz")
    np.savez_compressed(npz_path, A_list=A_list, B_list=B_list, C_list=C_list)
    print(f"Wrote {npz_path} (count={count}, M={M}, N={N}, K={K})")

    # Also write simple hex memory files for TB to load (flatten row-major)
    for t in range(count):
        a_hex = os.path.join(out_dir, f"A_{t}.hex")
        b_hex = os.path.join(out_dir, f"B_{t}.hex")
        c_hex = os.path.join(out_dir, f"C_{t}.hex")
        # write as signed 32-bit hex two's complement
        with open(a_hex, "w") as fa:
            for x in A_list[t].reshape(-1):
                fa.write(f"{(int(x) & 0xFFFFFFFF):08x}\n")
        with open(b_hex, "w") as fb:
            for x in B_list[t].reshape(-1):
                fb.write(f"{(int(x) & 0xFFFFFFFF):08x}\n")
        with open(c_hex, "w") as fc:
            for x in C_list[t].reshape(-1):
                fc.write(f"{(int(x) & 0xFFFFFFFF):08x}\n")
    print(f"Also wrote per-vector .hex files for TB in {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--out", type=str, default="tb/integration/test_vectors")
    args = p.parse_args()
    gen_vectors(args.m, args.n, args.k, args.count, args.out)
