import torch
import torch.nn as nn
from loss_functions import combined_loss

def evaluate(model, dataloader, device):
    """
    Evaluate the model on the given dataloader.
    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for validation/test data.
        device (torch.device): The device to use (CPU or GPU).
    Returns:
        float: The average loss on the dataset.
    """
    model.eval()
    criterion = nn.MSELoss()  # Use the same loss function as in training
    total_loss = 0
    total_mae = []

    with torch.no_grad():  # Disable gradient computation
        for graphs, speed, steer, angular, labels, seq_ids in dataloader:
            # Move data to the same device as the model
            graphs = graphs.to(device)
            speed = speed.to(device)
            steer = steer.to(device)
            angular = angular.to(device)
            labels = labels.to(device)
            seq_ids = seq_ids.to(device)

            # Forward pass
            predictions = model(graphs, speed, steer, angular, graphs.batch)

            # Compute loss
            loss, mae= combined_loss(predictions.squeeze(), labels)
            total_loss += loss.item()
            total_mae.append(mae)

    # Return the average loss
    return total_loss / len(dataloader), sum(total_mae) / len(total_mae)


def save_model(model, path):
    """Save the model state to a file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    """Load the model state from a file."""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")

def log_to_file(message, file_path="train_log.txt"):
    """Append a log message to a file."""
    with open(file_path, "a") as f:
        f.write(message + "\n")
