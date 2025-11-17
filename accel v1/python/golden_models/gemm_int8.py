"""gemm_int8.py - Golden GEMM checker for 2x2 systolic array test
Reads A_inputs.csv, B_inputs.csv, C_results.csv from tb/integration.
Computes C[i][j] = sum_k A[i][k]*B[k][j] in 32-bit signed math.
"""

import csv, sys, argparse
from collections import defaultdict


def load_a(path):
    A = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            Tk = int(r["Tk"])
            row = int(r["r"])
            k = int(r["k"])
            val = int(r["val"])
            A[Tk][row][k] = val
    return A


def load_b(path):
    B = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            Tk = int(r["Tk"])
            k = int(r["k"])
            col = int(r["c"])
            val = int(r["val"])
            B[Tk][k][col] = val
    return B


def load_c(path):
    C = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            Tk = int(r["Tk"])
            row = int(r["r"])
            col = int(r["c"])
            val = int(r["val"])
            C[Tk][row][col] = val
    return C


def check(A, B, C):
    fails = 0
    for Tk in sorted(C.keys()):
        rows = sorted(C[Tk].keys())
        cols = sorted({c for r in C[Tk].values() for c in r.keys()})
        for i in rows:
            for j in cols:
                acc = 0
                for k in range(Tk):
                    try:
                        a = A[Tk][i][k]
                        b = B[Tk][k][j]
                    except KeyError:
                        print(f"[Tk={Tk}] Missing operand i={i} j={j} k={k}")
                        fails += 1
                        break
                    acc += a * b
                rtl = C[Tk][i][j]
                if rtl != acc:
                    print(f"[FAIL] Tk={Tk} i={i} j={j} expected={acc} got={rtl}")
                    fails += 1
        if fails == 0:
            print(f"[PASS] Tk={Tk}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adir", default="tb/integration/A_inputs.csv")
    ap.add_argument("--bdir", default="tb/integration/B_inputs.csv")
    ap.add_argument("--cdir", default="tb/integration/C_results.csv")
    args = ap.parse_args()
    A = load_a(args.adir)
    B = load_b(args.bdir)
    C = load_c(args.cdir)
    fails = check(A, B, C)
    if fails == 0:
        print("gemm_int8: PASS")
        sys.exit(0)
    else:
        print("gemm_int8: FAIL count=%d" % fails)
        sys.exit(1)


if __name__ == "__main__":
    main()
