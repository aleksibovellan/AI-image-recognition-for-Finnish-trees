# AI/ML Trained Image Recognition for Finnish trees with a Web Interface
# Author: Aleksi Bovellan (2024)


# AI/ML IMAGE MODEL TRAINING SCRIPT FOR NEW INTELLIGENCE

# Requires a folder 'processed_trees', which is created by finishing the image pre-processing script beforehand.
# The resulted training intelligence from this script will be stored into a file "tree_species_model.pth".


# User-friendly settings: Define number of epochs, learning rate, and patience for early stopping

# Example: Setting an epoch value from "20" to "100" takes around 10 minutes on a year 2020 MacBook M1 Air
NUM_EPOCHS = 100  # Number of training epochs

LEARNING_RATE = 0.0010  # Fine-tuned learning rate (slower for better precision)
EARLY_STOPPING_PATIENCE = 15  # Stop training if no improvement for 15 epochs


# From here on the script will proceed automatically.

# Import necessary libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from torchvision.models import ResNet18_Weights

# Set up directories
PROCESSED_DATA_DIR = './processed_trees'  # Input folder with processed images

# Image transformations for training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}

# Ensure the processed data directory exists before proceeding
if not os.path.exists(PROCESSED_DATA_DIR):
    raise FileNotFoundError(f"\nError: Processed data directory '{PROCESSED_DATA_DIR}' not found. Please run the preprocessing script first.\n")

# Load datasets from the processed image folder
full_dataset = datasets.ImageFolder(PROCESSED_DATA_DIR, transform=data_transforms['train'])

# Split into 80% training and 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Data loaders for efficient batching and shuffling during training
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Print class distribution in training dataset
class_counts = np.bincount([label for _, label in train_dataset])
print("\nClass distribution in the training set:")
for i, count in enumerate(class_counts):
    print(f"Class {i} ({full_dataset.classes[i]}): {count} samples")

# Define ResNet18 model and modify final layer for 6 classes (tree species)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(full_dataset.classes))  # Output for 6 classes

# Loss function and optimizer with a fine-tuned learning rate
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)  # Fine-tuned learning rate

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Early stopping variables
best_val_loss = np.inf  # Initialize the best validation loss with infinity
epochs_no_improve = 0  # Counter for epochs with no improvement

# Training function with early stopping
def train_model(model, criterion, optimizer, num_epochs):
    """
    This function trains the model for the specified number of epochs. It includes early stopping.
    """
    global best_val_loss, epochs_no_improve

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            running_loss += loss.item()

        print(f'\nEpoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}')

        # After each epoch, evaluate the model
        val_loss = evaluate_model(model, return_loss=True)
        
        # Early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  # Reset patience counter
            print(f"Validation loss improved to {val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), 'tree_species_model.pth')  # Save the best model
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
        
        # Check if early stopping should trigger
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs - it seems learning has reached its peak.")
            break

# Evaluation function with loss calculation
def evaluate_model(model, return_loss=False):
    """
    This function evaluates the model on the validation dataset and returns the loss if specified.
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'\nValidation Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}')

    if return_loss:
        return val_loss / len(val_loader)  # Return the average validation loss
    else:
        return None

# Main execution
if __name__ == "__main__":
    print("\n--- Starting Training ---\n")
    train_model(model, criterion, optimizer, NUM_EPOCHS)  # Train with early stopping
    print("\n--- Training Completed ---\n")
