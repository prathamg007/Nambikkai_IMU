import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Utility conversions
# ------------------------------------------------------------

def quat_to_euler(q):
    """Convert quaternion array (N,4) -> Euler angles (deg): roll, pitch, yaw."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ysqr = y * y

    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    roll_x = np.degrees(np.arctan2(t0, t1))

    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, +1.0)
    pitch_y = np.degrees(np.arcsin(t2))

    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    yaw_z = np.degrees(np.arctan2(t3, t4))

    return np.vstack([roll_x, pitch_y, yaw_z]).T  # (N,3)


def quat_to_axis_angle(q):
    """Compute axis-angle magnitude (deg) from quaternions."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    angles = 2 * np.degrees(np.arccos(np.clip(q[:, 0], -1.0, 1.0)))
    return angles


def unwrap_deg(a):
    # unwrap in radians, return degrees
    return np.rad2deg(np.unwrap(np.deg2rad(a)))


# ------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------

def plot_euler_comparison(pred_csv, save_path=None):
    """
    Plot ground-truth vs predicted Euler angles over time.
    """
    df = pd.read_csv(pred_csv)
    q_true = df[["QW_true", "QX_true", "QY_true", "QZ_true"]].values
    q_pred = df[["QW_pred", "QX_pred", "QY_pred", "QZ_pred"]].values

    e_true = quat_to_euler(q_true)
    e_pred = quat_to_euler(q_pred)

    for i in range(3):
        e_true[:, i] = unwrap_deg(e_true[:, i])
        e_pred[:, i] = unwrap_deg(e_pred[:, i])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    titles = ["Roll (°)", "Pitch (°)", "Yaw (°)"]
    for i in range(3):
        axes[i].plot(e_true[:, i], label="Ground Truth", color="black", linewidth=1)
        axes[i].plot(e_pred[:, i], label="Predicted", color="orange", linestyle="--")
        axes[i].set_ylabel(titles[i])
        axes[i].grid(True)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Sample index (window center)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def plot_axis_angle_comparison(pred_csv, save_path=None):
    """
    Plot ground-truth vs predicted axis-angle magnitudes (deg).
    """
    df = pd.read_csv(pred_csv)
    q_true = df[["QW_true", "QX_true", "QY_true", "QZ_true"]].values
    q_pred = df[["QW_pred", "QX_pred", "QY_pred", "QZ_pred"]].values

    a_true = quat_to_axis_angle(q_true)
    a_pred = quat_to_axis_angle(q_pred)

    plt.figure(figsize=(10, 4))
    plt.plot(a_true, label="Ground Truth Axis Angle", color="black", linewidth=1)
    plt.plot(a_pred, label="Predicted Axis Angle", color="orange", linestyle="--")
    plt.ylabel("Axis Angle (°)")
    plt.xlabel("Sample index (window center)")
    plt.title("Axis-Angle Magnitude Comparison")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def plot_quaternion_components(pred_csv, save_path=None):
    """
    Plot all quaternion components (QW,QX,QY,QZ) true vs predicted.
    """
    df = pd.read_csv(pred_csv)
    q_true = df[["QW_true", "QX_true", "QY_true", "QZ_true"]].values
    q_pred = df[["QW_pred", "QX_pred", "QY_pred", "QZ_pred"]].values

    labels = ["QW", "QX", "QY", "QZ"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for i, lbl in enumerate(labels):
        axes[i].plot(q_true[:, i], label=f"{lbl}_True", color="black", linewidth=1)
        axes[i].plot(q_pred[:, i], label=f"{lbl}_Pred", color="orange", linestyle="--")
        axes[i].set_ylabel(lbl)
        axes[i].grid(True)
        if i == 0:
            axes[i].legend()
    axes[-1].set_xlabel("Sample index (window center)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def plot_quaternion_error(pred_csv, save_path=None):
    """
    Plot instantaneous quaternion angular error (deg) over time.
    """
    df = pd.read_csv(pred_csv)
    if "ang_error_deg" not in df.columns:
        raise ValueError("Column 'ang_error_deg' missing. Ensure eval.py saved predicted CSV with errors.")
    plt.figure(figsize=(10, 4))
    plt.plot(df["ang_error_deg"], color="crimson", linewidth=0.8)
    plt.title("Instantaneous Quaternion Angular Error (°)")
    plt.xlabel("Sample index (window center)")
    plt.ylabel("Error (°)")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


# ------------------------------------------------------------
# Master convenience wrapper (run all plots together)
# ------------------------------------------------------------
def generate_all_plots(pred_csv, output_prefix="logs/plots"):
    """
    Generates:
      - Euler angle comparison (3 subplots)
      - Axis-angle comparison
      - Quaternion components (4 subplots)
      - Angular error vs time
    """
    plot_euler_comparison(pred_csv, save_path=f"{output_prefix}_euler.png")
    plot_axis_angle_comparison(pred_csv, save_path=f"{output_prefix}_axisangle.png")
    plot_quaternion_components(pred_csv, save_path=f"{output_prefix}_quats.png")
    plot_quaternion_error(pred_csv, save_path=f"{output_prefix}_error.png")
    print(f"✅ All plots saved with prefix {output_prefix}_*.png")
