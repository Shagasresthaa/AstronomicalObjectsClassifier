import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from src.data_prep.preprocess import convert_to_numpy, data_split, clear_split, split_data, get_loaders
from src.evaluation.eval import write_test_eval
import tensorboard

batch_size = 32  # Adjust as needed
resplit_data = True
epochs = 1 # Adjust number of epochs
lr = 0.001  # Adjust learning rate



astroImages = "noFilter"
zoomFilter = "zoomFilter"
invertedFilter = "invFilter"
masks = "OBFQFilter"

data_dir = "data"
processed = "/processed"
raw = "/raw/image_extracts"

current_dataset = zoomFilter

processed_dir = data_dir + processed + current_dataset
raw_dir = data_dir + raw + current_dataset

model_name = "Zoom_1epoch"
class_names = ['galaxy', 'qso', 'star']


start = time.time()
if data_split(processed_dir, class_names) and resplit_data:
    clear_split(processed_dir, class_names)

if not data_split(processed_dir, class_names) or resplit_data:
    split_data(processed_dir, raw_dir, class_names)

    print(f"Data split in {time.time() - start:.2f} seconds")

train_loader, test_loader, val_loader = get_loaders(processed_dir, batch_size)

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input channels = 3 for RGB
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten output of conv layers
            nn.Linear(64 * 32 * 32, 128),  # Adjust based on your image size
            nn.ReLU(),
            nn.Linear(128, 3)  # Output classes = 3
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

model = MyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

log_file = open(f'logs/log_{time.strftime("%Y%m%d")}_{model_name}.txt', 'w')
log_file.write(f'Epochs: {epochs}\n')
log_file.write(f'Batch size: {batch_size}\n')
log_file.write(f'Optimizer: Adam\n')
log_file.write(f'Learning rate: {lr}\n')
log_file.write(f'Loss function: CrossEntropyLoss\n')
log_file.write(f'Epoch: Step: Loss:\n')

#start timer
start = time.time()
#use tqdm to show progress bar
for epoch in tqdm(range(epochs)):
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:  # Log progress every 100 mini-batches
            log_file.write(f'{epoch + 1}/{epochs}, {i + 1}, {loss.item():.4f}\n')

# format time in hours, minutes, seconds
end_time = time.time() - start
hours, rem = divmod(end_time, 3600)
minutes, seconds = divmod(rem, 60)

time_str = "Training completed in {:.0f} hours, {:.0f} minutes, {:.0f} seconds".format(hours, minutes, seconds)
print(time_str)
log_file.write(time_str)
log_file.close()
#save model, naming it using date and name
torch.save(model.state_dict(), f'models/model_{time.strftime("%Y%m%d")}_{model_name}.pt')

log_file_dir = f'logs/log_{time.strftime("%Y%m%d")}_{model_name}.txt'
write_test_eval(test_loader, model, device, log_file_dir)