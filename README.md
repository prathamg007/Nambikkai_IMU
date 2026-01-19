# Nambikkai IMU: Deep Learning Based 6-DoF Motion Capture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

**Nambikkai IMU** is a robust deep learning framework designed to reconstruct high-fidelity 3D trajectories from low-cost MEMS inertial sensors. By leveraging a **Bi-directional LSTM (Bi-LSTM)** architecture, this system effectively mitigates the sensor drift and noise artifacts common in traditional integration-based tracking methods.

This project was developed as a core component for a medical rehabilitation wearable, enabling precise remote diagnosis of patient limb movements.

## 🚀 Key Features

* **Deep Sensor Fusion:** Replaces standard Extended Kalman Filters (EKF) with a data-driven Bi-LSTM model to map raw inertial data to global orientation.
* **Drift Reduction:** Achieves significant reduction in orientation drift compared to double integration methods.
* **Quaternion Output:** Predicts 4D Quaternion vectors directly to avoid Gimbal lock issues associated with Euler angles.
* **Modular Pipeline:**
    * **Config-Driven:** Fully parameterizable via `config.yaml`.
    * **Custom Data Loaders:** Efficient handling of sliding window time-series data.
    * **Visualization Tools:** Built-in utilities to plot predicted vs. ground truth trajectories (Roll/Pitch/Yaw).

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/prathamg007/Nambikkai_IMU.git](https://github.com/prathamg007/Nambikkai_IMU.git)
    cd Nambikkai_IMU
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Project Structure

├── config.yaml # Hyperparameters (Batch size, LR, Window size) 
├── model.py # Bi-LSTM Neural Network Architecture 
├── train.py # Main training loop 
├── eval.py # Inference and evaluation script 
├── dataset_loader_custom.py # PyTorch Dataset for windowed IMU data 
├── data_windowing.py # Preprocessing: Slicing raw data into windows 
├── loss_utils.py # Custom Quaternion Loss functions 
└── plot_utils.py # Visualization tools for trajectories


## ⚙️ Configuration (`config.yaml`)

Control the training dynamics by editing `config.yaml`:

```yaml
training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100

data:
  window_size: 200   # Number of time-steps per sample
  stride: 10         # Overlap between windows
  input_dim: 6       # 3 Accel + 3 Gyro
```

  🏃 Usage
1. Training the Model
To train the model on your dataset, run:
python train.py --config config.yaml

This will save the best model weights to the checkpoints/ directory.

2. Evaluation & Testing
To evaluate the model on unseen data and visualize the results:
python eval.py --model_path checkpoints/best_model.pth --input data/test_sequence.csv

This will generate trajectory plots comparing the Predicted Orientation vs. Ground Truth.

🧠 Model Architecture
The core model is a Recurrent Neural Network (RNN) specifically designed for sequence-to-sequence regression:

