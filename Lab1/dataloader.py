import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import random

def get_train_val_loaders(data_dir, batch_size=32):
    # Define data augmentation and transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the full training dataset
    full_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
    
    # Get indices for all classes
    targets = np.array(full_dataset.targets)
    class_to_indices = {class_idx: np.where(targets == class_idx)[0] for class_idx in range(len(full_dataset.classes))}

    train_indices = []
    val_indices = []

    # Randomly select 5 samples from each class for the validation set
    for class_idx, indices in class_to_indices.items():
        indices = list(indices)
        random.shuffle(indices)  # Shuffle the indices randomly
        val_indices.extend(indices[:5])  # Select the first 5 as validation set
        train_indices.extend(indices[5:])  # The rest go to the training set

    # Create Subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Print the number of samples in each class in the validation set
    print_class_distribution(val_dataset, full_dataset)

    return train_loader, val_loader

def print_class_distribution(val_dataset, full_dataset):
    # Initialize a dictionary to count the number of samples per class
    class_counts = {class_name: 0 for class_name in full_dataset.classes}

    # Iterate through the validation set samples and count
    for idx in val_dataset.indices:
        label = full_dataset.targets[idx]
        class_name = full_dataset.classes[label]
        class_counts[class_name] += 1

    # Print the number of samples in each class
    print("Class distribution in validation set:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

def get_test_loader(data_dir, batch_size=32):
    # Define data augmentation and transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the test dataset
    test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=transform)

    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
