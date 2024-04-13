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
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure the log directory exists
log_directory = "logs/model_logs"
os.makedirs(log_directory, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
starConfMatrixTimeFolder = timestamp
log_filename = f"star_class_classifier_cnn_{timestamp}.log"
log_filepath = os.path.join(log_directory, log_filename)

logging.basicConfig(filename=log_filepath, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s', filemode='w')

# Class encoding dictionary
class_encoding = {'A': 0, 'F': 1, 'G': 2, 'K': 3, 'M': 4}

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
        features = torch.tensor(star_data.values, dtype=torch.float).transpose(0, 1)
        star_class = self.master_dataframe.iloc[idx]['StarClass']
        label = class_encoding[star_class]
        target = torch.tensor(label, dtype=torch.long)
        return features, target

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.branch3x3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.branch5x5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        
        self.branch_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(out_channels * 4)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch3x3 = self.branch3x3(x)
        
        branch5x5 = self.branch5x5(x)
        
        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return self.batch_norm(self.relu(torch.cat(outputs, 1)))  # Concatenate in the depth dimension

class StarCNN(nn.Module):
    def __init__(self, sequence_length):
        super(StarCNN, self).__init__()
        self.inception1 = InceptionModule(in_channels=2, out_channels=32)
        self.inception2 = InceptionModule(in_channels=128, out_channels=64)
        reduced_sequence_length = sequence_length // 4
        
        # The inception module concatenates 4 branches, hence we multiply out_channels by 4
        flattened_size = reduced_sequence_length * 64 * 4
        self.fc = nn.Linear(flattened_size, 5)  # Adjust for the number of classes

    def forward(self, x):
        x = self.inception1(x)
        x = F.max_pool1d(x, kernel_size=2)  # Cutting the sequence length by 2
        x = self.inception2(x)
        x = F.max_pool1d(x, kernel_size=2)  # Cutting the sequence length by 2
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def evaluate_model(model, dataloader, criterion, device, phase="Testing"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100 * correct / total
    logging.info(f'{phase} Loss: {total_loss / len(dataloader):.4f}, {phase} Accuracy: {accuracy:.2f}%')
    
    if phase == "Testing":
        cm = confusion_matrix(all_targets, all_preds)
        report = classification_report(all_targets, all_preds, target_names=list(class_encoding.keys()))
        logging.info(f"{phase} Classification Report:\n{report}")
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(class_encoding.keys()), yticklabels=list(class_encoding.keys()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()  # Adjust layout to fit all labels
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plotFolder = f'notebooks/starClassifierModel/star_classifier_analysis_{starConfMatrixTimeFolder}'
        os.makedirs(plotFolder, exist_ok=True)
        plt.savefig(f'{plotFolder}/{phase}_confusion_matrix_{timestamp}.png')
        plt.close()  # Close the plot to free memory

    return total_loss / len(dataloader), accuracy

def train_and_evaluate_model(model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler, epochs=25, patience=5):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, targets in train_dataloader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_accuracy = 100 * train_correct / train_total
        logging.info(f'Epoch {epoch+1}, Training Loss: {train_loss / len(train_dataloader):.4f}, Training Accuracy: {train_accuracy:.2f}%')
        
        # Validation Phase
        val_loss, val_accuracy = evaluate_model(model, val_dataloader, criterion, device, "Testing")
        scheduler.step(val_loss)  # Adjust the learning rate based on the validation loss

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model_path = f'models/star_model_{starConfMatrixTimeFolder}'
            best_model_path = f'{model_path}/best_fold_model_epoch_{epoch+1}.pt'
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'Best model saved at {best_model_path} for this fold')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f'Early stopping triggered after {epoch+1} epochs')
                break

    return best_val_loss
def trainModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load the dataset
    master_file_path = 'data/augmentation/modelData/starNormalizedData/masterDataFile.csv'
    master_data = pd.read_csv(master_file_path)

    # Set up K-Fold cross-validation
    k_folds = 10
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Initialize results list for storing fold results
    results = []

    # Iterate over each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(master_data)):
        logging.info(f'Starting fold {fold+1}')

        # Split data into training and validation for the current fold
        train_subset = master_data.iloc[train_idx]
        val_subset = master_data.iloc[val_idx]

        train_dataset = StarDataset(train_subset)
        val_dataset = StarDataset(val_subset)

        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=4)

        # Initialize model, criterion, optimizer, and scheduler
        model = StarCNN(sequence_length=4642).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        # Train and evaluate model on this fold
        fold_loss = train_and_evaluate_model(model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler, epochs=25, patience=5)
        results.append(fold_loss)

        logging.info(f'Fold {fold+1} completed with loss {fold_loss:.4f}')

    # Log the average loss across all folds
    average_loss = sum(results) / len(results)
    logging.info(f'Average Loss across folds: {average_loss:.4f}')

if __name__ == "__main__":
    trainModel()