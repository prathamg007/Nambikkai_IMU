import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class ColumnMap:
    common_ts_ms: str = "common_ts_ms"
    acc_ts: str = "ACC_ts"
    ax: str = "AX"
    ay: str = "AY"
    az: str = "AZ"
    gyr_ts: str = "GYR_ts"
    gx: str = "GX"
    gy: str = "GY"
    gz: str = "GZ"
    quat_ts: str = "QUAT_ts"
    qw: str = "QW"
    qx: str = "QX"
    qy: str = "QY"
    qz: str = "QZ"

def _assert_columns(df: pd.DataFrame, cm: ColumnMap):
    needed = [cm.common_ts_ms, cm.acc_ts, cm.ax, cm.ay, cm.az,
              cm.gyr_ts, cm.gx, cm.gy, cm.gz,
              cm.quat_ts, cm.qw, cm.qx, cm.qy, cm.qz]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

def _interp_1d(t_src: np.ndarray, x_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    return np.interp(t_dst, t_src, x_src)

def _interp_nd(t_src: np.ndarray, X_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    if X_src.ndim == 1:
        return _interp_1d(t_src, X_src, t_dst)
    out = np.empty((t_dst.shape[0], X_src.shape[1]), dtype=np.float64)
    for k in range(X_src.shape[1]):
        out[:, k] = _interp_1d(t_src, X_src[:, k], t_dst)
    return out

def _normalize_quat(Q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(Q, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return Q / n

def _clean_time_series(t_ms: np.ndarray, X: np.ndarray):
    idx = np.argsort(t_ms, kind="mergesort")
    t = t_ms[idx].astype(np.float64)
    Xs = X[idx]
    uniq, pos = np.unique(t, return_index=True)
    return uniq, Xs[pos]

def load_and_interpolate_csv(csv_path: str, columns: Dict[str, str], resample_hz: Optional[float] = None, trim_seconds: float = 0.0):
    df = pd.read_csv(csv_path)
    cm = ColumnMap(**columns)
    _assert_columns(df, cm)

    t_common_ms = df[cm.common_ts_ms].to_numpy(dtype=np.int64)
    t_acc_ms    = df[cm.acc_ts].to_numpy(dtype=np.int64); acc  = df[[cm.ax, cm.ay, cm.az]].to_numpy(dtype=np.float64)
    t_gyr_ms    = df[cm.gyr_ts].to_numpy(dtype=np.int64); gyr  = df[[cm.gx, cm.gy, cm.gz]].to_numpy(dtype=np.float64)
    t_q_ms      = df[cm.quat_ts].to_numpy(dtype=np.int64); quat = df[[cm.qw, cm.qx, cm.qy, cm.qz]].to_numpy(dtype=np.float64)

    t_common_ms, _ = _clean_time_series(t_common_ms, t_common_ms)
    t_acc_ms, acc  = _clean_time_series(t_acc_ms, acc)
    t_gyr_ms, gyr  = _clean_time_series(t_gyr_ms, gyr)
    t_q_ms,   quat = _clean_time_series(t_q_ms,   quat)

    t0 = max(t_acc_ms.min(), t_gyr_ms.min(), t_q_ms.min(), t_common_ms.min())
    t1 = min(t_acc_ms.max(), t_gyr_ms.max(), t_q_ms.max(), t_common_ms.max())
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        print("\n[DEBUG] Invalid file:", csv_path)
        print("  ACC timestamps:", df[columns["acc_ts"]].min(), "→", df[columns["acc_ts"]].max())
        print("  GYR timestamps:", df[columns["gyr_ts"]].min(), "→", df[columns["gyr_ts"]].max())
        print("  QUAT timestamps:", df[columns["quat_ts"]].min(), "→", df[columns["quat_ts"]].max())
        raise ValueError("Non-overlapping or invalid time ranges among streams.")

    if resample_hz is None:
        diffs = np.diff(t_common_ms).astype(np.float64)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            diffs = np.diff(t_acc_ms).astype(np.float64); diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            diffs = np.diff(t_gyr_ms).astype(np.float64); diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            raise ValueError("Cannot infer sampling interval; timestamps not valid.")
        dt_ms = np.mean(diffs)
        resample_hz = 1000.0 / dt_ms
    dt_ms = 1000.0 / float(resample_hz)

    t_uniform_ms = np.arange(t0, t1 + 0.5*dt_ms, dt_ms, dtype=np.float64)

    acc_i  = _interp_nd(t_acc_ms, acc,  t_uniform_ms)
    gyr_i  = _interp_nd(t_gyr_ms, gyr,  t_uniform_ms)
    quat_i = _interp_nd(t_q_ms,   quat, t_uniform_ms)
    quat_i = _normalize_quat(quat_i)

    if trim_seconds and trim_seconds > 0:
        trim_ms = float(trim_seconds) * 1000.0
        lo, hi = t0 + trim_ms, t1 - trim_ms
        mask = (t_uniform_ms >= lo) & (t_uniform_ms <= hi)
        if mask.sum() < 10:
            raise ValueError("Trim removed too much data; reduce trim_seconds.")
        acc_i, gyr_i, quat_i, t_uniform_ms = acc_i[mask], gyr_i[mask], quat_i[mask], t_uniform_ms[mask]

    fs = 1000.0 / float(np.mean(np.diff(t_uniform_ms)))
    t_uniform_s = t_uniform_ms / 1000.0

    for arr in (acc_i, gyr_i, quat_i):
        arr[~np.isfinite(arr)] = 0.0

    return acc_i, gyr_i, quat_i, t_uniform_s, float(fs), df
