from datetime import datetime
import logging
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import shuffle
import numpy as np

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
classEncoding = {'O': 0, 'B': 1, 'A': 2, 'F': 3, 'G': 4, 'K': 5, 'M': 6}

class StarDataset(Dataset):
    """ Dataset class for loading data """
    def __init__(self, master_dataframe):
        self.master_dataframe = master_dataframe

    def __len__(self):
        return len(self.master_dataframe)

    def __getitem__(self, idx):
        file_path = self.master_dataframe.iloc[idx]['normalizedAndPaddedDataPath']
        star_data = pd.read_csv(file_path)[['normalized_wavelength', 'normalized_smoothed_flux']]
        features = torch.tensor(star_data.values, dtype=torch.float).transpose(0, 1)
        star_class = self.master_dataframe.iloc[idx]['starClass']
        label = classEncoding[star_class]
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
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        outputs = self.dropout(torch.cat(outputs, 1))
        return self.batch_norm(self.relu(outputs))

class StarCNN(nn.Module):
    def __init__(self, sequence_length):
        super(StarCNN, self).__init__()
        self.inception1 = InceptionModule(in_channels=2, out_channels=32)
        self.inception2 = InceptionModule(in_channels=128, out_channels=64)
        self.inception3 = InceptionModule(in_channels=256, out_channels=128)
        self.inception4 = InceptionModule(in_channels=512, out_channels=256)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.3)

        reduced_sequence_length = sequence_length // (2**4)  

        flattened_size = reduced_sequence_length * 256 * 4
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 7)  
        
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.inception1(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.inception2(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.inception3(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = self.inception4(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def evaluate_model(model, dataloader, criterion, device, phase="Testing"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    #misclassified_examples = []  # To store information about misclassified examples

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

            ## Check for misclassified examples
            # No longer being used
            #mismatches = predicted != targets
            #misclassified_indices = torch.nonzero(mismatches, as_tuple=False).squeeze(1).cpu().numpy()
            #if misclassified_indices.ndim == 0:  # Single element
            #    misclassified_indices = [misclassified_indices.item()]
            #misclassified_examples.extend([
            #    (index, pred.item(), true.item()) 
            #    for index, pred, true in zip(misclassified_indices, predicted[mismatches], targets[mismatches])
            #])

    accuracy = 100 * correct / total
    logging.info(f'{phase} Loss: {total_loss / len(dataloader):.4f}, {phase} Accuracy: {accuracy:.2f}%')

    if phase == "Testing":
        cm = confusion_matrix(all_targets, all_preds)
        report = classification_report(all_targets, all_preds, target_names=list(classEncoding.keys()))
        logging.info(f"{phase} Classification Report:\n{report}")

        # Log the misclassified examples details for error analysis
        #for example in misclassified_examples:
            #idx, pred, true = example
            #logging.info(f'Misclassified Example: Index {idx}, Predicted Class {pred}, True Class {true}')

        # Plotting the confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(classEncoding.keys()), yticklabels=list(classEncoding.keys()))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()  # Adjust layout to fit all labels
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plotFolder = f'modelPerformanceMetrics/starCNNClassifier/star_classifier_analysis_{starConfMatrixTimeFolder}/confusionMatrix'
        os.makedirs(plotFolder, exist_ok=True)
        plt.savefig(f'{plotFolder}/{phase}_confusion_matrix_{timestamp}.png')
        plt.close()  # Close the plot to free memory

    return total_loss / len(dataloader), accuracy, np.array(all_targets), np.array(all_preds)


def plot_metrics(metrics, fold, plot_directory, metric_name):
    plt.figure(figsize=(10, 5))
    for key, values in metrics.items():
        plt.plot(values, label=f'{key}')
    plt.title(f'{metric_name} - Fold {fold + 1}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plot_directory}/{metric_name}_fold_{fold + 1}.png')
    plt.close()

def plot_precision_recall(precision, recall, fold, plot_directory):
    plt.figure(figsize=(10, 5))
    plt.scatter(recall, precision, c=np.linspace(0, 1, len(precision)))
    plt.colorbar(label='Epochs')
    plt.title(f'Precision vs Recall - Fold {fold + 1}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig(f'{plot_directory}/Precision_vs_Recall_fold_{fold + 1}.png')
    plt.close()

def train_and_evaluate_model(model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler, epochs=30, patience=6):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    epoch_metrics = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': [],
        'precision': [],
        'recall': []
    }

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

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
        val_loss, val_accuracy, all_targets, all_preds = evaluate_model(model, val_dataloader, criterion, device, "Testing")
        scheduler.step(val_loss)

        # Log Learning Rate
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch+1}, Current Learning Rate: {current_lr}')

        # Store metrics
        epoch_metrics['train_loss'].append(train_loss / len(train_dataloader))
        epoch_metrics['train_accuracy'].append(train_accuracy)
        epoch_metrics['val_loss'].append(val_loss)
        epoch_metrics['val_accuracy'].append(val_accuracy)
        epoch_metrics['learning_rate'].append(current_lr)

        # Calculate and store precision and recall
        precision, recall, _, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
        epoch_metrics['precision'].append(precision)
        epoch_metrics['recall'].append(recall)

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

    return epoch_metrics

def generateCombinedMasterFile(sampleSizes):
    # Load and concatenate the DataFrames
    class_files = {
        'data/augmentation/modelTrainingData/masterFileData/O_class_master_data_stage_4.csv': sampleSizes.get('O', 2896),
        'data/augmentation/modelTrainingData/masterFileData/B_class_master_data_stage_4.csv': sampleSizes.get('B', 4000),
        'data/augmentation/modelTrainingData/masterFileData/A_class_master_data_stage_4.csv': sampleSizes.get('A', 4000),
        'data/augmentation/modelTrainingData/masterFileData/F_class_master_data_stage_4.csv': sampleSizes.get('F', 4000),
        'data/augmentation/modelTrainingData/masterFileData/G_class_master_data_stage_4.csv': sampleSizes.get('G', 4000),
        'data/augmentation/modelTrainingData/masterFileData/K_class_master_data_stage_4.csv': sampleSizes.get('K', 4000),
        'data/augmentation/modelTrainingData/masterFileData/M_class_master_data_stage_4.csv': sampleSizes.get('M', 4000)
    }

    # Use a generator expression to read specified number of rows for each file
    masterDataFiles = pd.concat(
        (pd.read_csv(filename, nrows=nrows) for filename, nrows in class_files.items()),
        ignore_index=True
    )

    # Shuffle the combined DataFrame
    masterDataFiles = shuffle(masterDataFiles, random_state=42)
    masterDataFiles.reset_index(drop=True, inplace=True)
    masterDataFiles = shuffle(masterDataFiles, random_state=72)
    masterDataFiles = masterDataFiles[['objid','plate','mjd','fiberid', 'fitsFilePath','fitsExtractCSVInitPath','denoisedCSVDataPath','augmentedCSVDataPath','normalizedAndPaddedDataPath','columnSize','lambdaMax','lambdaMaxMeters','starClass','wavelengthMin','wavelengthMax','smoothedFluxMin','smoothedFluxMax']]
    # Save or further process the shuffled DataFrame
    masterDataFiles.to_csv('data/augmentation/modelTrainingData/combined_master_star_classes_shuffled.csv', index=False)

def trainModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load and shuffle the dataset
    # Adjust the sample sizes in the dict
    # Except for class O which has 2896 samples in its entirity, all other classes are limited to max of 5000 datapoints unless you have downloaded and processed more data
    # and normalized and ready to be consumed in direcrtory "modelTrainingData"
    # If you need more data first make sure you have the extract of all star data (if not available run newClassDataFetch under data_fetch)
    # Once you have star data run script isolateTodoStarSpectraDownload under data_fetch which will see what all data you already have and what you need to download
    # Once that script isolates and generates files, adjust dict in script missingDataFetch under data_fetch to specify how many samples you want for each class and run script
    # Once that is done run script starClassDataPreprocessAndAugment under data_prep which will check for corrupt fits files redownload, extract, augment, normalize and pad the data
    # Once this is completed verify the master files under data>augmentation>modelTrainingData>masterFileData
    # Once this is done you can proceed with running the model as all data is prepped and ready
    sampleSizes = {'O': 2896, 'B': 4000, 'A': 4000, 'F': 4000, 'G': 4000, 'K': 4000, 'M': 4000}
    generateCombinedMasterFile(sampleSizes)
    master_file_path = 'data/augmentation/modelTrainingData/combined_master_star_classes_shuffled.csv'
    master_data = pd.read_csv(master_file_path)
    master_data = shuffle(master_data, random_state=42)  # Re-Shuffling the data (Just to be sure)
    seqLen = max(master_data['columnSize'])

    # Set up K-Fold cross-validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(master_data)):
        logging.info(f'Starting fold {fold+1}')
        train_subset = master_data.iloc[train_idx]
        val_subset = master_data.iloc[val_idx]

        # Calculate class weights for the current fold based on training data
        class_counts = train_subset['starClass'].value_counts()
        total_samples = len(train_subset)
        class_weights = {class_id: total_samples / count for class_id, count in class_counts.items()}
        # Manually adjust weights for specific classes
        class_weights['B'] *= 1.5  
        class_weights['F'] *= 1.5  
        class_weights['G'] *= 1.5  
        class_weights['M'] *= 0.8  
        weights = torch.tensor([class_weights.get(class_id, 1.0) for class_id in sorted(class_weights)], dtype=torch.float32).to(device)
        
        # Log class distribution in the subsets
        train_class_counts = train_subset['starClass'].value_counts().to_dict()
        val_class_counts = val_subset['starClass'].value_counts().to_dict()
        logging.info(f'Train class counts for fold {fold+1}: {train_class_counts}')
        logging.info(f'Validation class counts for fold {fold+1}: {val_class_counts}')

        train_dataset = StarDataset(train_subset)
        val_dataset = StarDataset(val_subset)
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16)

        # Initialize model, criterion, optimizer, and scheduler
        model = StarCNN(sequence_length=seqLen).to(device)  # Ensure sequence length matches your data
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        epoch_metrics = train_and_evaluate_model(model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler)
        
        # Folder for saving plots
        plot_folder = f'modelPerformanceMetrics/starCNNClassifier/star_classifier_analysis_{starConfMatrixTimeFolder}/fold_{fold+1}'
        os.makedirs(plot_folder, exist_ok=True)

        # Plot metrics
        plot_metrics({'Learning Rate': epoch_metrics['learning_rate']}, fold, plot_folder, 'Learning Rate')
        plot_metrics({'Train Accuracy': epoch_metrics['train_accuracy'], 'Val Accuracy': epoch_metrics['val_accuracy']}, fold, plot_folder, 'Accuracy')
        plot_metrics({'Learning Rate': epoch_metrics['learning_rate']}, fold, plot_folder, 'Learning Rate')
        plot_precision_recall(epoch_metrics['precision'], epoch_metrics['recall'], fold, plot_folder)

if __name__ == "__main__":
    trainModel()

