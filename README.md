# Federated Skeleton-based Action Recognition

This project provides a **proof-of-concept implementation** of a Federated Learning (FL) system for **human action recognition from skeleton data**.  
The main goal is to demonstrate a **privacy-preserving training approach** where models can learn collaboratively from decentralized data **without raw data ever leaving the client’s device**.

The system is lightweight and runs without requiring large dataset downloads or complex setups.  
It highlights the **core challenges of FL**, particularly training on **non-identically distributed (Non-IID)** data.

---

## 🚀 Core Concepts Demonstrated

- **Federated Learning**  
  Simulates a central server that aggregates model updates from multiple clients, producing a global model collaboratively.

- **Data Privacy**  
  Raw skeleton data never leaves the client. Only model updates are shared.

- **Non-IID Data Distribution**  
  Client datasets are split unevenly using a Dirichlet distribution, reflecting the reality that different users perform different actions.

- **Skeleton-based Action Recognition**  
  Uses an **LSTM neural network** to classify sequences of 2D skeleton keypoints into human actions (e.g., *waving*, *jumping*).

- **Real-time Inference Pipeline**  
  Includes a script that performs **live webcam-based action recognition** using MediaPipe for pose estimation.

---

## 📂 File Structure

- **`federated_skeleton_learning.py`**  
  Core training script:
  - Generates synthetic skeleton dataset  
  - Simulates federated learning with multiple clients  
  - Trains an `ActionRecognitionModel` with Federated Averaging (FedAvg)  
  - Saves global model → `action_recognition_model.pth`

- **`predict_action.py`**  
  Real-time inference script:
  - Loads `action_recognition_model.pth`  
  - Captures webcam feed using OpenCV  
  - Extracts skeleton keypoints via MediaPipe  
  - Classifies actions in real-time and overlays results on video  

---

## ⚙️ Setup & Installation

1. Clone this repository or download the files:
   ```bash
   git clone https://github.com/amey0307/Federated-Skeleton-based-Action-Recognition.git
   cd Federated-Skeleton-based-Action-Recognition

2.	(Optional) Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate    # macOS/Linux
    venv\Scripts\activate       # Windows

3.	Install dependencies:
    ```bash
    pip install torch numpy opencv-python mediapipe