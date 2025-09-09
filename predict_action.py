import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# --- Re-define the Model Architecture from your training script ---
# This class MUST be identical to the one in federated_skeleton_learning.py
class ActionRecognitionModel(nn.Module):
    """A simple LSTM model for classifying skeleton action sequences."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ActionRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1, :, :])
        out = self.log_softmax(out)
        return out

# --- Configuration ---
# These should match the settings used during training
CONFIG = {
    "num_keypoints": 15, # We will select 15 keypoints from MediaPipe
    "keypoint_dim": 2,
    "sequence_length": 40,
    "actions": ["waving", "jumping", "clapping"],
}
MODEL_PATH = "action_recognition_model.pth" # Path where the trained model will be saved/loaded

# --- Helper Function to Save a Dummy Model ---
def save_dummy_trained_model():
    """
    In a real scenario, you would load the model trained by your federated script.
    For demonstration, we'll create and save a new, untrained model instance.
    You should replace this with your actual trained model file.
    """
    print(f"Creating a dummy model at '{MODEL_PATH}' for demonstration purposes.")
    input_size = CONFIG["num_keypoints"] * CONFIG["keypoint_dim"]
    model = ActionRecognitionModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=len(CONFIG["actions"])
    )
    # In a real use case, you would load the state_dict from your trained model.
    # For now, we just save the initialized weights.
    torch.save(model.state_dict(), MODEL_PATH)
    print("Dummy model saved.")

# --- Main Prediction Logic ---
def predict_real_time():
    """
    Captures webcam feed, performs pose estimation, and predicts actions in real-time.
    """
    # 1. Load the trained model
    try:
        model_state_dict = torch.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}.")
        save_dummy_trained_model()
        model_state_dict = torch.load(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = CONFIG["num_keypoints"] * CONFIG["keypoint_dim"]
    model = ActionRecognitionModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=len(CONFIG["actions"])
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    print("Action recognition model loaded successfully.")

    # 2. Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # 3. Setup Webcam Capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # 4. Real-time processing loop
    keypoints_sequence = deque(maxlen=CONFIG["sequence_length"])
    current_prediction = "..."

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find pose
        results = pose.process(image_rgb)

        # Extract and draw landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # --- Data Preprocessing for the Model ---
            # Select a subset of keypoints to match the model's input
            landmarks = results.pose_landmarks.landmark
            selected_keypoints = np.array([
                [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y],
                [landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y],
            ]).flatten() # Flatten to a 1D array of size 30 (15*2)

            keypoints_sequence.append(selected_keypoints)

            # --- Prediction ---
            if len(keypoints_sequence) == CONFIG["sequence_length"]:
                # We have a full sequence, time to predict
                sequence_tensor = torch.tensor(np.array(keypoints_sequence), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(sequence_tensor)
                    _, predicted_idx = torch.max(outputs.data, 1)
                    current_prediction = CONFIG["actions"][predicted_idx.item()]

        # Display the prediction on the screen
        cv2.putText(image, f"Action: {current_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the image
        cv2.imshow('Real-time Action Recognition', image)

        if cv2.waitKey(5) & 0xFF == 27: # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == '__main__':
    # NOTE: You need to modify your training script to save the final model.
    # Add this line at the end of the main() function in federated_skeleton_learning.py:
    # torch.save(global_model.state_dict(), "action_recognition_model.pth")
    #
    # Then run the training script first to generate the model file.
    predict_real_time()