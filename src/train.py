import os, argparse, yaml, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
import glob, os

from .dataset_loader_custom import load_and_interpolate_csv
from .data_windowing import create_bidirectional_windows
from .model import build_modelA
from .loss_utils import quaternion_loss_QME, angular_error_deg

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(cfg_path: str):
    cfg = load_cfg(cfg_path)

    csv_entry = cfg["data"]["csv_files"]

    if isinstance(csv_entry, str):
        # Case 1: user gave a folder path like "data/processed"
        if os.path.isdir(csv_entry):
            csv_files = sorted(glob.glob(os.path.join(csv_entry, "*.csv")))
            if not csv_files:
                raise RuntimeError(f"[ERROR] No CSV files found in folder: {csv_entry}")
        # Case 2: user gave a wildcard path like "data/processed/train_*.csv"
        elif "*" in csv_entry:
            csv_files = sorted(glob.glob(csv_entry))
            if not csv_files:
                raise RuntimeError(f"[ERROR] No CSV files matched pattern: {csv_entry}")
        # Case 3: user gave a single CSV file path
        elif os.path.isfile(csv_entry):
            csv_files = [csv_entry]
        else:
            raise ValueError(f"[ERROR] Invalid csv_files entry: {csv_entry}")
    elif isinstance(csv_entry, list):
        # Case 4: user provided explicit list of files
        csv_files = csv_entry
    else:
        raise ValueError("[ERROR] data.csv_files must be a path string, wildcard, or list of paths.")

    print(f"[INFO] Found {len(csv_files)} CSV files for training:")
    for f in csv_files:
        print("   ", f)

    columns     = cfg["data"]["columns"]
    resample_hz = cfg["data"]["resample_hz"]
    trim_s      = cfg["data"]["trim_seconds"]

    W = int(cfg["windowing"]["window_size"])
    S = int(cfg["windowing"]["stride"])

    bs  = int(cfg["training"]["batch_size"])
    ep  = int(cfg["training"]["epochs"])
    vs  = float(cfg["training"]["validation_split"])
    opt_name = str(cfg["training"]["optimizer"]).lower()
    lr  = float(cfg["training"]["learning_rate"])
    wd  = float(cfg["training"]["weight_decay"])
    name= str(cfg["training"]["model_name"])

    Xg_all, Xa_all, F_all, Y_all = [], [], [], []
    for csv in csv_files:
        try:
            acc, gyr, quat, t_s, fs, _ = load_and_interpolate_csv(csv, columns, resample_hz, trim_s)
        except ValueError as e:
            print(f"[WARN] Skipping {csv}: {e}")
            continue
        except KeyError as e:
            print(f"[WARN] Missing column {e} in {csv}, skipping.")
            continue
        except Exception as e:
            print(f"[WARN] Unexpected error in {csv}: {e}")
            continue

        # Skip if quaternion timestamps are constant or empty
        try:
            qts = quat[:, 0] if hasattr(quat, '__getitem__') else []
            if isinstance(qts, (list, tuple)) and len(set(qts)) <= 1:
                print(f"[WARN] Skipping {csv}: constant quaternion timestamps or values.")
                continue
        except Exception:
            pass

        Xg, Xa, F, Y = create_bidirectional_windows(acc, gyr, quat, fs, W, S)
        if Xg.shape[0] == 0:
            print(f"[WARN] No windows generated for {csv}; skipping.")
            continue

        Xg_all.append(Xg)
        Xa_all.append(Xa)
        F_all.append(F)
        Y_all.append(Y)

        if Xg.shape[0] == 0:
            print(f"[WARN] No windows generated for {csv}; skipping.")
            continue
        Xg_all.append(Xg); Xa_all.append(Xa); F_all.append(F); Y_all.append(Y)

    if not Xg_all:
        raise RuntimeError("No training data windows produced.")

    Xg = np.concatenate(Xg_all, axis=0)
    Xa = np.concatenate(Xa_all, axis=0)
    F  = np.concatenate(F_all,  axis=0)
    Y  = np.concatenate(Y_all,  axis=0)

    model = build_modelA(W)

    if opt_name == "adamw":
        import tensorflow_addons as tfa
        opt = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt, loss=quaternion_loss_QME, metrics=[angular_error_deg])

    os.makedirs("saved_models", exist_ok=True); os.makedirs("logs", exist_ok=True)
    ckpt = os.path.join("saved_models", f"{name}_best.h5")
    callbacks = [
        ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=int(cfg["callbacks"]["early_stopping_patience"]), restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=float(cfg["callbacks"]["reduce_lr_factor"]), patience=int(cfg["callbacks"]["reduce_lr_patience"])),
        CSVLogger(os.path.join("logs", f"{name}_train.csv")),
        TensorBoard(log_dir=os.path.join("logs", f"tb_{name}"))
    ]

    model.fit([Xg, Xa, F], Y, batch_size=bs, epochs=ep, validation_split=vs, shuffle=True, callbacks=callbacks, verbose=1)

    model.save(os.path.join("saved_models", f"{name}_final.h5"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
