from PIL import Image
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#take path and class names, and create numpy files for each class
def convert_to_numpy(data_dir, class_names):
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        file_names = os.listdir(class_dir)
        images = np.zeros((len(file_names), 128, 128, 3), dtype=np.uint8)
        for i, file_name in enumerate(file_names):
            image = Image.open(os.path.join(class_dir, file_name))
            images[i] = np.array(image)
        np.save(os.path.join(data_dir, f'{class_name}.npy'), images)
    return 1

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
def split_data(processed_dir, raw_dir, class_names):

    for split in ['train', 'test', 'val']:
        for class_name in class_names:
            os.makedirs(os.path.join(processed_dir, split, class_name), exist_ok=True)

    for class_name in class_names:
        source_dir = os.path.join(raw_dir, class_name)
        file_names = os.listdir(source_dir)
        t = tqdm(total=len(train_files) + len(test_files) + len(val_files), desc=f'Splitting data...')
        # Split file names into train, test, and val sets
        train_files, test_files = train_test_split(
            file_names, test_size=0.2, random_state=42, stratify= [class_name] * len(file_names)
        )
        train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42, stratify= [class_name] * len(train_files))
        for file_name in train_files:
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(processed_dir, 'train', class_name))
            t.update(1)
        for file_name in test_files:
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(processed_dir, 'test', class_name))
            t.update(1)
        for file_name in val_files:
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(processed_dir, 'val', class_name))
            t.update(1)
    return 1




def get_loaders(processed_dir, batch_size):
    class ImageDataset(datasets.ImageFolder):
        def __init__(self, root, transform=None):
            super().__init__(root, transform=transform)

        def __getitem__(self, index):
            path, target = self.samples[index]
            image = Image.open(path).convert('RGB')  # Ensure RGB format

            if self.transform is not None:
                image = self.transform(image)

            return image, target
    data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),  # Converts PIL Image to PyTorch Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    train_dataset = ImageDataset(root=processed_dir + '/train', transform=data_transform)
    test_dataset = ImageDataset(root= processed_dir + '/test', transform=data_transform) 
    val_dataset = ImageDataset(root= processed_dir + '/val', transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader