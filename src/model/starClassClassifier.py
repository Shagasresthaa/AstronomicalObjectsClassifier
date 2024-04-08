from datetime import datetime
import logging
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Logging setup with timestamped filenames
log_directory = "logs/model_logs"

# Ensure the log directory exists
os.makedirs(log_directory, exist_ok=True)  
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"star_class_classifier_cnn_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s', filemode='w')

# Class encoding dictionary
class_encoding = {'A': 0, 'F': 1, 'G': 2, 'K': 3, 'M': 4}

def evaluate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    total_correct = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for features, targets in dataloader:
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total_correct += (predicted == targets).sum().item()
            all_preds.extend(predicted.numpy())
            all_targets.extend(targets.numpy())
    
    # Calculate overall metrics
    accuracy = total_correct / len(dataloader.dataset)
    logging.info(f'Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    # Confusion Matrix and Classification Report
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=list(class_encoding.keys()))
    logging.info("Classification Report:\n", report)
    
    # Plotting confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(class_encoding.keys()), yticklabels=list(class_encoding.keys()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Save the plot to a file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fp = 'notebooks/starClassifierModel/'
    os.makedirs(fp, exist_ok=True)
    plt.savefig(f'{fp}/model_{timestamp}.png')

class StarDataset(Dataset):
    def __init__(self, master_dataframe):
        self.master_dataframe = master_dataframe

    def __len__(self):
        return len(self.master_dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.master_dataframe.iloc[idx]['NormalizedFilePath']
        star_data = pd.read_csv(file_path)[['normalized_wavelength', 'normalized_smoothed_flux']]
        features = torch.tensor(star_data.values, dtype=torch.float).transpose(0, 1)  # Transpose to [2, sequence_length]

        # Encoding the star class
        star_class = self.master_dataframe.iloc[idx]['StarClass']
        label = class_encoding[star_class]  # Convert class to integer label
        target = torch.tensor(label, dtype=torch.long)

        return features, target

class StarCNN(nn.Module):
    def __init__(self, sequence_length):
        super(StarCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * (sequence_length // 4), 5)  # Adjusted for 5 output classes

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def main():
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load and prepare data
    master_file_path = 'data/augmentation/modelData/starNormalizedData/masterDataFile.csv'
    master_data = pd.read_csv(master_file_path)

    # Split data into train and test sets
    train_data, test_data = train_test_split(master_data, test_size=0.2, random_state=42)

    # Create datasets and dataloaders for training and testing
    train_dataset = StarDataset(train_data)
    test_dataset = StarDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)  # No need to shuffle the test data

    # Model, loss function, and optimizer
    model = StarCNN(sequence_length=4642)  # Adjust sequence_length as needed
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(15):  # Number of epochs
        for features, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        logging.info(f'Epoch {epoch+1}, Training Loss: {loss.item()}')

    # Evaluation
    evaluate_model(model, test_dataloader, criterion)

if __name__ == "__main__":
    main()