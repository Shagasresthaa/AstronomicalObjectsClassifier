import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import time
batch_size = 32  # Adjust as needed
resplit_data = True
epochs = 15  # Adjust number of epochs
lr = 0.001  # Adjust learning rate

data_dir = "data"
processed_dir = data_dir + "/processed/astroImages"
raw_dir = data_dir + "/raw/image_extracts/astroImages"
model_name = "default_32fBatchSize"
class_names = ['galaxy', 'qso', 'star']

#check if processed directory exists, with images in each class
def data_split(processed_dir, class_names):
    if os.path.exists(os.path.join(processed_dir, 'train', class_names[0])):
        if len(os.listdir(os.path.join(processed_dir, 'train', class_names[0]))):
            print("Data already processed")
            return 1
    return 0

def clear_split(processed_dir, class_names):
    print("Resplitting data...")
    for split in ['train', 'test', 'val']:
        for class_name in class_names:
            shutil.rmtree(os.path.join(processed_dir, split, class_name))
    return 1

#split data into train, test, and validation sets
def split_data(processed_dir, class_names):

    for split in ['train', 'test', 'val']:
        for class_name in class_names:
            os.makedirs(os.path.join(processed_dir, split, class_name), exist_ok=True)

    for class_name in tqdm(class_names):
        source_dir = os.path.join(raw_dir, class_name)
        file_names = os.listdir(source_dir)

        # Split file names into train, test, and val sets
        train_files, test_files = train_test_split(
            file_names, test_size=0.2, random_state=42, stratify= [class_name] * len(file_names)
        )
        train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42, stratify= [class_name] * len(train_files))
        for file_name in train_files:
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(processed_dir, 'train', class_name))
        for file_name in test_files:
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(processed_dir, 'test', class_name))
        for file_name in val_files:
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(processed_dir, 'val', class_name))
    return 1

start = time.time()

if data_split(processed_dir, class_names) and resplit_data:
    clear_split(processed_dir, class_names)
if not data_split(processed_dir, class_names) or resplit_data:
    split_data(processed_dir, class_names)

    print(f"Data split in {time.time() - start:.2f} seconds")


data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # Converts PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

class ImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')  # Ensure RGB format

        if self.transform is not None:
            image = self.transform(image)

        return image, target

train_dataset = ImageDataset(root=processed_dir + '/train', transform=data_transform)
test_dataset = ImageDataset(root= processed_dir + '/test', transform=data_transform) 
val_dataset = ImageDataset(root= processed_dir + '/val', transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
epochs = 10  # Adjust number of epochs
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

#for true positives
true_stars = 0
true_galaxies = 0
true_qso = 0

#for false positives
false_stars = 0
false_galaxies = 0
false_qso = 0

#for missed classifications
missed_stars = 0
missed_galaxies = 0
missed_qso = 0

false_star_galaxy = 0
false_star_qso = 0
false_galaxy_star = 0
false_galaxy_qso = 0
false_qso_star = 0
false_qso_galaxy = 0

total_stars = 0
total_galaxies = 0
total_qso = 0

#test model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        
        avg_galaxy_confidence = torch.mean(probabilities[:, 0]).item()
        avg_qso_confidence = torch.mean(probabilities[:, 1]).item()
        avg_star_confidence = torch.mean(probabilities[:, 2]).item()

        total += labels.size(0)
        total_stars += labels[labels == 0].size(0)
        total_galaxies += labels[labels == 1].size(0)
        total_qso += labels[labels == 2].size(0)
        correct += (predicted == labels).sum().item()

        true_stars += ((predicted == 0) & (labels == 0)).sum().item()
        true_galaxies += ((predicted == 1) & (labels == 1)).sum().item()
        true_qso += ((predicted == 2) & (labels == 2)).sum().item()

        false_stars += ((predicted == 0) & (labels != 0)).sum().item()
        false_galaxies += ((predicted == 1) & (labels != 1)).sum().item()
        false_qso += ((predicted == 2) & (labels != 2)).sum().item()

        missed_stars += ((predicted != 0) & (labels == 0)).sum().item()
        missed_galaxies += ((predicted != 1) & (labels == 1)).sum().item()
        missed_qso += ((predicted != 2) & (labels == 2)).sum().item()

        false_star_galaxy += ((predicted == 0) & (labels == 1)).sum().item()
        false_star_qso += ((predicted == 0) & (labels == 2)).sum().item()
        false_galaxy_star += ((predicted == 1) & (labels == 0)).sum().item()
        false_galaxy_qso += ((predicted == 1) & (labels == 2)).sum().item()
        false_qso_star += ((predicted == 2) & (labels == 0)).sum().item()
        false_qso_galaxy += ((predicted == 2) & (labels == 1)).sum().item()



accuracy = correct / total
print(f'Accuracy on test set: {accuracy * 100:.2f}%')
log_file.write(f'Accuracy on test set: {accuracy * 100:.2f}%\n')
# class accuracy percentages
# total_stars = true_stars + missed_stars
# total_galaxies = true_galaxies + missed_galaxies
# total_qso = true_qso + missed_qso
true_stars = true_stars / total_stars
false_stars = false_stars / total_stars
missed_stars = missed_stars / total_stars
true_galaxies = true_galaxies / total_galaxies
false_galaxies = false_galaxies / total_galaxies
missed_galaxies = missed_galaxies / total_galaxies
true_qso = true_qso / total_qso
false_qso = false_qso / total_qso
missed_qso = missed_qso / total_qso

false_star_galaxy = false_star_galaxy / total_galaxies
false_star_qso = false_star_qso / total_qso
false_galaxy_star = false_galaxy_star / total_stars
false_galaxy_qso = false_galaxy_qso / total_qso
false_qso_star = false_qso_star / total_stars
false_qso_galaxy = false_qso_galaxy / total_galaxies



log_file.write(f'True stars: {true_stars * 100:.2f}%\n')
log_file.write(f'False stars: {false_stars * 100:.2f}%\n')
log_file.write(f'Missed stars: {missed_stars * 100:.2f}%\n\n')
log_file.write(f'True galaxies: {true_galaxies * 100:.2f}%\n')
log_file.write(f'False galaxies: {false_galaxies * 100:.2f}%\n')
log_file.write(f'Missed galaxies: {missed_galaxies * 100:.2f}%\n\n')
log_file.write(f'True qso: {true_qso * 100:.2f}%\n')
log_file.write(f'False qso: {false_qso * 100:.2f}%\n')
log_file.write(f'Missed qso: {missed_qso * 100:.2f}%\n')

log_file.write(f'Galaxies classified as stars: {false_star_galaxy * 100:.2f}%\n')
log_file.write(f'QSO classified as stars: {false_star_qso * 100:.2f}%\n')
log_file.write(f'Stars classified as galaxies: {false_galaxy_star * 100:.2f}%\n')
log_file.write(f'QSO classified as galaxies: {false_galaxy_qso * 100:.2f}%\n')
log_file.write(f'Stars classified as QSO: {false_qso_star * 100:.2f}%\n')
log_file.write(f'Galaxies classified as QSO: {false_qso_galaxy * 100:.2f}%\n')

log_file.write(f'Average star confidence: {avg_star_confidence:.2f}\n')
log_file.write(f'Average galaxy confidence: {avg_galaxy_confidence:.2f}\n')
log_file.write(f'Average qso confidence: {avg_qso_confidence:.2f}\n')

log_file.close()

#save model, naming it using date and name
torch.save(model.state_dict(), f'models/model_{time.strftime("%Y%m%d")}_{model_name}.pt')
