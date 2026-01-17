import os, argparse, yaml, json, numpy as np, tensorflow as tf
from tensorflow.keras.models import load_model
from .dataset_loader_custom import load_and_interpolate_csv
from .data_windowing import create_bidirectional_windows
from .loss_utils import quaternion_loss_QME, angular_error_deg
import pandas as pd

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def evaluate(model_path: str, csv_path: str, columns, resample_hz, trim_s, W, S, out_json: str):
    custom_objects = {"quaternion_loss_QME": quaternion_loss_QME, "angular_error_deg": angular_error_deg}
    model = load_model(model_path, custom_objects=custom_objects)

    acc, gyr, quat, t_s, fs, _ = load_and_interpolate_csv(csv_path, columns, resample_hz, trim_s)
    Xg, Xa, F, Y = create_bidirectional_windows(acc, gyr, quat, fs, W, S)

    Yp = model.predict([Xg, Xa, F], verbose=0)

    Yp = Yp / np.linalg.norm(Yp, axis=1, keepdims=True)
    Y  =  Y / np.linalg.norm(Y,  axis=1, keepdims=True)

    # --- hemisphere alignment: flip preds so dot ≥ 0 ---
    sign = np.sign(np.sum(Yp * Y, axis=1, keepdims=True))
    sign[sign == 0] = 1.0
    Yp = Yp * sign

    dot = np.sum(Yp * Y, axis=-1); dot = np.clip(np.abs(dot), 0.0, 1.0)
    ang = 2.0 * np.arccos(dot) * (180.0 / np.pi)

    stats = {"mean_deg": float(np.mean(ang)), "median_deg": float(np.median(ang)), "p90_deg": float(np.percentile(ang, 90)), "count": int(ang.shape[0])}
    if out_json:
        with open(out_json, "w") as f: json.dump(stats, f, indent=2)
    print("Angular Error (deg):", stats)
    df_out = pd.DataFrame({
    "QW_true": Y[:, 0],
    "QX_true": Y[:, 1],
    "QY_true": Y[:, 2],
    "QZ_true": Y[:, 3],
    "QW_pred": Yp[:, 0],
    "QX_pred": Yp[:, 1],
    "QY_pred": Yp[:, 2],
    "QZ_pred": Yp[:, 3],
    "ang_error_deg": ang
    })
    out_csv = out_json.replace(".json", "_preds.csv") if out_json else "predicted_quaternions.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved predicted vs ground-truth quaternions to {out_csv}")
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    W = int(cfg["windowing"]["window_size"]); S = int(cfg["windowing"]["stride"])
    columns = cfg["data"]["columns"]; resample_hz = cfg["data"]["resample_hz"]; trim_s = cfg["data"]["trim_seconds"]

    best = os.path.join("saved_models", f"{cfg['training']['model_name']}_best.h5")
    model_path = best if os.path.exists(best) else os.path.join("saved_models", f"{cfg['training']['model_name']}_final.h5")
    evaluate(model_path, args.csv, columns, resample_hz, trim_s, W, S, args.out)
