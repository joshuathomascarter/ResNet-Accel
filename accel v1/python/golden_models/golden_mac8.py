"""golden_mac8.py - Golden checker for mac8 testbench CSV
Reads tb/unit/mac8_results.csv produced by tb_mac8.v and reproduces the INT8xINT8
multiply-accumulate into a 32-bit signed accumulator with optional saturation.
"""

import csv, sys, argparse

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


def to_s32(x):
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def run(csv_path, sat=True):
    errs = 0
    acc = 0
    last_cycle = -1
    with open(csv_path, newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            cycle = int(row["cycle"])
            a = int(row["a"])
            b = int(row["b"])
            en = int(row["en"])
            clr = int(row["clr"])
            acc_rtl = int(row["acc"])
            sat_flag_rtl = int(row["sat"])

            if clr:
                acc = 0
            if en:
                prod = a * b
                sum_raw = acc + prod
                sum_wrapped = to_s32(sum_raw)
                pos_oflow = acc >= 0 and prod >= 0 and sum_wrapped < 0
                neg_oflow = acc < 0 and prod < 0 and sum_wrapped >= 0
                if sat:
                    if pos_oflow:
                        acc = INT32_MAX
                    elif neg_oflow:
                        acc = INT32_MIN
                    else:
                        acc = sum_wrapped
                else:
                    acc = sum_wrapped
                exp_sat = 1 if (pos_oflow or neg_oflow) else 0
            else:
                exp_sat = 0
            if acc_rtl != acc:
                print(f"ACC MISMATCH cycle={cycle} got={acc_rtl} exp={acc}")
                errs += 1
            if sat_flag_rtl != exp_sat:
                print(f"SAT_FLAG MISMATCH cycle={cycle} got={sat_flag_rtl} exp={exp_sat}")
                errs += 1
            last_cycle = cycle
    if errs == 0:
        print("golden_mac8: PASS")
    else:
        print(f"golden_mac8: FAIL errors={errs}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="tb/unit/mac8_results.csv")
    ap.add_argument("--sat", type=int, default=1)
    args = ap.parse_args()
    rc = run(args.csv, sat=bool(args.sat))
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
