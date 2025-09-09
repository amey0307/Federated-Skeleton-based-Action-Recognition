import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import copy
from collections import defaultdict

# --- 1. Configuration and Hyperparameters ---
CONFIG = {
    "num_clients": 3,
    "num_rounds": 15,
    "clients_per_round": 2,
    "local_epochs": 5,
    "batch_size": 10,
    "learning_rate": 0.01,
    "num_keypoints": 15,
    "keypoint_dim": 2, # Using 2D skeletons (x, y)
    "sequence_length": 40, # 40 frames per action
    "num_samples_per_action": 100,
    "actions": ["waving", "jumping", "clapping"],
    "non_iid_alpha": 0.5, # Lower alpha means more Non-IID
    "model_path": "action_recognition_model.pth" # Define model path here
}

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- 2. Synthetic Skeleton Data Generation ---
# This section creates a simple dataset on the fly, avoiding large downloads.
# The data mimics skeleton movements for different actions.

def generate_synthetic_data():
    """Generates a dataset of synthetic skeleton sequences for different actions."""
    print("Generating synthetic skeleton data...")
    num_actions = len(CONFIG["actions"])
    data = []
    labels = []

    for action_idx, action_name in enumerate(CONFIG["actions"]):
        for _ in range(CONFIG["num_samples_per_action"]):
            # Start with a base skeleton pose (e.g., standing)
            base_pose = np.random.rand(CONFIG["num_keypoints"], CONFIG["keypoint_dim"]) * 0.5 + 0.25
            sequence = np.zeros((CONFIG["sequence_length"], CONFIG["num_keypoints"], CONFIG["keypoint_dim"]))

            for frame_idx in range(CONFIG["sequence_length"]):
                frame = base_pose.copy()
                t = frame_idx / CONFIG["sequence_length"] # Time factor [0, 1]
                noise = np.random.normal(0, 0.01, frame.shape) # Add some noise

                if action_name == "waving":
                    # Animate right hand (keypoint 8) up and down
                    frame[8, 1] += 0.2 * np.sin(t * 4 * np.pi) # y-coordinate
                elif action_name == "jumping":
                    # Move the whole body up and down
                    frame[:, 1] += 0.15 * abs(np.sin(t * 2 * np.pi))
                elif action_name == "clapping":
                     # Move hands (keypoints 8 and 9) towards each other
                    frame[8, 0] += 0.1 * np.sin(t * 4 * np.pi) # x-coordinate of right hand
                    frame[9, 0] -= 0.1 * np.sin(t * 4 * np.pi) # x-coordinate of left hand

                sequence[frame_idx] = frame + noise

            data.append(sequence.reshape(CONFIG["sequence_length"], -1)) # Flatten keypoints
            labels.append(action_idx)

    print(f"Data generation complete. Total samples: {len(data)}")
    return torch.tensor(np.array(data), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

class SkeletonDataset(Dataset):
    """Custom PyTorch Dataset for our synthetic skeleton data."""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- 3. Model Architecture ---
# An LSTM-based model suitable for sequence classification tasks like action recognition.

class ActionRecognitionModel(nn.Module):
    """A simple LSTM model for classifying skeleton action sequences."""
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ActionRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # LSTM output is (output, (hidden_state, cell_state))
        # We only need the output of the last time step
        _, (h_n, _) = self.lstm(x)
        # h_n is of shape (num_layers, batch, hidden_size), we take the last layer
        out = self.fc(h_n[-1, :, :])
        out = self.log_softmax(out)
        return out

# --- 4. Federated Learning Simulation ---
# This section contains the logic for the server and clients,
# simulating the federated training process.

def create_non_iid_split(dataset, num_clients, alpha):
    """
    Splits data among clients in a non-IID fashion using a Dirichlet distribution.
    This simulates a more realistic federated scenario where clients have different data distributions.
    """
    # The 'dataset' is a torch.utils.data.Subset object after random_split.
    # We need to access the labels from the underlying original dataset using the subset's indices.
    labels = dataset.dataset.labels[dataset.indices].numpy()
    num_classes = len(np.unique(labels))
    client_data_indices = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Ensure proportions sum to 1 and assign data points
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        client_data_indices = [
            cli_idx + part.tolist()
            for cli_idx, part in zip(client_data_indices, np.split(idx_k, proportions))
        ]

    client_datasets = []
    for indices in client_data_indices:
        client_data = torch.utils.data.Subset(dataset, indices)
        client_datasets.append(client_data)

    # Print data distribution for verification
    print("\nClient Data Distribution (Non-IID):")
    for i, client_dataset in enumerate(client_datasets):
        label_counts = defaultdict(int)
        # For a Subset, we get the label by accessing the item, which returns (data, label)
        client_labels = [dataset[idx][1] for idx in client_dataset.indices]
        for label in client_labels:
            label_counts[CONFIG["actions"][label.item()]] += 1
        print(f"  Client {i+1}: {dict(sorted(label_counts.items()))}")

    return client_datasets


def client_update(client_loader, model, device):
    """Simulates a single client's training process."""
    local_model = copy.deepcopy(model)
    local_model.to(device)
    local_model.train()

    optimizer = optim.Adam(local_model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.NLLLoss()

    for _ in range(CONFIG["local_epochs"]):
        for data, labels in client_loader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    return local_model.state_dict()


def federated_averaging(models_weights):
    """Aggregates client model weights to update the global model."""
    if not models_weights:
        return None

    # Initialize a new state_dict with zeros
    avg_weights = copy.deepcopy(models_weights[0])
    for key in avg_weights.keys():
        avg_weights[key] = torch.zeros_like(avg_weights[key])

    # Sum up all the weights
    for weights in models_weights:
        for key in avg_weights.keys():
            avg_weights[key] += weights[key]

    # Divide by the number of models to get the average
    for key in avg_weights.keys():
        avg_weights[key] = torch.div(avg_weights[key], len(models_weights))

    return avg_weights

def evaluate_model(model, test_loader, device):
    """Evaluates the performance of the global model on the test set."""
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# --- 5. Main Execution Block ---

def main():
    """Main function to run the federated learning simulation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 1. Prepare Data
    data, labels = generate_synthetic_data()
    full_dataset = SkeletonDataset(data, labels)

    # Split data into training and testing
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # Distribute training data among clients
    client_datasets = create_non_iid_split(train_dataset, CONFIG["num_clients"], CONFIG["non_iid_alpha"])
    client_loaders = [DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=True) for ds in client_datasets]

    # 2. Initialize Global Model
    input_size = CONFIG["num_keypoints"] * CONFIG["keypoint_dim"]
    global_model = ActionRecognitionModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        num_classes=len(CONFIG["actions"])
    ).to(device)

    print("\n--- Starting Federated Training ---")
    # 3. Start Federated Training Rounds
    for round_num in range(CONFIG["num_rounds"]):
        # Select a subset of clients for this round
        selected_clients_indices = random.sample(range(CONFIG["num_clients"]), CONFIG["clients_per_round"])
        client_weights = []

        print(f"\nRound {round_num + 1}/{CONFIG['num_rounds']} | Participating clients: {[i+1 for i in selected_clients_indices]}")

        # Dispatch model to clients and train
        for client_idx in selected_clients_indices:
            local_weights = client_update(client_loaders[client_idx], global_model, device)
            client_weights.append(local_weights)

        # Aggregate client updates
        global_weights = federated_averaging(client_weights)
        if global_weights:
            global_model.load_state_dict(global_weights)

        # Evaluate the global model's performance
        accuracy = evaluate_model(global_model, test_loader, device)
        print(f"Round {round_num + 1} | Global Model Accuracy on Test Set: {accuracy:.2f}%")

    print("\n--- Federated Training Finished ---")
    final_accuracy = evaluate_model(global_model, test_loader, device)
    print(f"\nFinal Global Model Accuracy: {final_accuracy:.2f}%")

    # Save the final model
    print(f"Saving final trained model to '{CONFIG['model_path']}'")
    torch.save(global_model.state_dict(), CONFIG['model_path'])


if __name__ == "__main__":
    main()

