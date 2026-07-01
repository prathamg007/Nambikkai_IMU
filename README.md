# Nambikkai IMU — Deep Learning 6-DoF Motion Capture

Deep-learning attitude estimation for a **wearable rehabilitation device**: the system predicts unit-quaternion limb orientation directly from raw 6-DoF IMU windows (3-axis accelerometer + 3-axis gyroscope), replacing drift-prone integration and costly optical motion-capture for remote physiotherapy assessment.

Built as a Project Intern at **Nambikkai Pvt. Ltd., Chennai** (Aug–Oct 2025).

## How it works

1. **Sensor synchronization** (`src/dataset_loader_custom.py`) — accelerometer, gyroscope, and reference-quaternion streams arrive with independent timestamps and rates. The loader sorts and de-duplicates each stream, finds their common time overlap, infers the sampling rate, resamples everything onto a uniform grid by linear interpolation, and re-normalizes quaternions.
2. **Windowing** (`src/data_windowing.py`) — odd-length, center-labelled sliding windows (default **101 samples, stride 10**) give the network symmetric temporal context around the labelled sample.
3. **Model A** (`src/model.py`) — dual-branch **CNN + Bi-LSTM**: each sensor branch stacks two `GaussianNoise → Conv1D(128, k=11) → Mish → MaxPool(3) → Dropout` blocks followed by a `BiLSTM(128)`; branch embeddings are concatenated and fused with a **sampling-rate input** so one model serves devices logging at different rates. A final `Dense(4)` + L2-normalization emits a valid unit quaternion. Architecture adapted from Golroudbari & Sabour's end-to-end inertial attitude estimation framework (see `End-to-End-Deep-Learning-Framework.../CITATION.cff`), re-engineered for this custom dual-probe wearable dataset.
4. **Loss** (`src/loss_utils.py`) — **Quaternion Multiplicative Error**: the error quaternion `q_true ⊗ conj(q_pred)` is computed with Hamilton products and the L1 norm of its vector part is minimized — a manifold-aware objective, with angular error (degrees) tracked as the metric.
5. **Evaluation** (`src/eval.py`) — hemisphere alignment resolves the quaternion double cover (q ≡ −q) before scoring; reports mean / median / p90 angular error and dumps per-window predictions to CSV.

## Results (held-out recordings)

| Session | Mean err | Median | p90 | Windows |
|---------|----------|--------|-----|---------|
| test_1  | **0.46°** | 0.46° | 0.50° | 182 |
| test_4  | 2.00°    | 1.79°  | 3.45° | 313 |
| test_2  | 3.73°    | 3.63°  | 5.11° | 184 |

Per-session plots (Euler angles, quaternions, axis-angle error) are in `logs/`.

## Usage

```bash
pip install -r requirements.txt

# train (config-driven; expects processed CSVs under data/processed)
python -m src.train --config configs/config.yaml

# evaluate a held-out recording
python -m src.eval --config configs/config.yaml --csv data/test/test_1.csv --out logs/test_1.json
```

> `src/` uses package-relative imports — run with `python -m src.train`, not `python src/train.py`.

Training defaults (`configs/config.yaml`): Adam, LR 5e-4, batch 256, ≤60 epochs, 15% validation split, early stopping + ReduceLROnPlateau, TensorBoard + CSV logging. Best/final weights land in `saved_models/`.

## Data

The wearable logs **two IMU probes per capture** (P1/P2). `data/script.py` splits raw dual-probe exports into unified per-probe training CSVs (`common_ts_ms, ACC_ts, AX..AZ, GYR_ts, GX..GZ, QUAT_ts, QW..QZ`); 125+ processed recordings were used for training. Sample held-out captures live in `data/test/`.

## Repo layout

```text
src/                  own pipeline: loader, windowing, model, QME loss, train, eval, plots
configs/config.yaml   all hyperparameters
data/                 dual-probe splitter + held-out test CSVs
logs/                 training curves, TensorBoard runs, per-session evaluation plots/JSON
saved_models/         best + final .h5 checkpoints
End-to-End-.../       reference framework (Golroudbari & Sabour) incl. classical AHRS baselines
```
