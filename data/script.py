import os
import pandas as pd

INPUT_DIR = "temp"
OUTPUT_DIR = "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

counter = 127  # you already have train_1..train_126

for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith(".csv"):
        continue

    fpath = os.path.join(INPUT_DIR, fname)
    df = pd.read_csv(fpath)

    cols = list(df.columns)

    # Detect RAW (your new layout): presence of P1_RAW_AX is enough
    is_raw = "P1_RAW_AX" in cols or "P2_RAW_AX" in cols

    if is_raw:
        # ===== RAW format: QUAT block appears earlier in the file =====
        p1 = df[[
            "common_ts_ms",
            "P1_RAW_ACC_ts", "P1_RAW_AX", "P1_RAW_AY", "P1_RAW_AZ",
            "P1_RAW_GYR_ts", "P1_RAW_GX", "P1_RAW_GY", "P1_RAW_GZ",
            "P1_QUAT_ts", "P1_QW", "P1_QX", "P1_QY", "P1_QZ",
        ]].copy()

        p2 = df[[
            "common_ts_ms",
            "P2_RAW_ACC_ts", "P2_RAW_AX", "P2_RAW_AY", "P2_RAW_AZ",
            "P2_RAW_GYR_ts", "P2_RAW_GX", "P2_RAW_GY", "P2_RAW_GZ",
            "P2_QUAT_ts", "P2_QW", "P2_QX", "P2_QY", "P2_QZ",
        ]].copy()
    else:
        # ===== Old format =====
        p1 = df[[
            "common_ts_ms",
            "P1_ACC_ts", "P1_AX", "P1_AY", "P1_AZ",
            "P1_GYR_ts", "P1_GX", "P1_GY", "P1_GZ",
            "P1_QUAT_ts", "P1_QW", "P1_QX", "P1_QY", "P1_QZ",
        ]].copy()

        p2 = df[[
            "common_ts_ms",
            "P2_ACC_ts", "P2_AX", "P2_AY", "P2_AZ",
            "P2_GYR_ts", "P2_GX", "P2_GY", "P2_GZ",
            "P2_QUAT_ts", "P2_QW", "P2_QX", "P2_QY", "P2_QZ",
        ]].copy()

    # Rename to unified headers in the EXACT required order
    unified_cols = [
        "common_ts_ms", "ACC_ts", "AX", "AY", "AZ",
        "GYR_ts", "GX", "GY", "GZ",
        "QUAT_ts", "QW", "QX", "QY", "QZ"
    ]
    p1.columns = unified_cols
    p2.columns = unified_cols

    # Save as train_127, train_128, ...
    p1.to_csv(os.path.join(OUTPUT_DIR, f"train_{counter}.csv"), index=False)
    counter += 1
    p2.to_csv(os.path.join(OUTPUT_DIR, f"train_{counter}.csv"), index=False)
    counter += 1

    print(f"OK: {fname} -> train_{counter-2}.csv, train_{counter-1}.csv")
