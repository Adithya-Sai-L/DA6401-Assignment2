from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode, RandomHorizontalFlip, RandomRotation
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def get_training_dataloaders(data_dir, batch_size=32, num_workers=4, augmentation=False):
    """
    Returns train and validation DataLoaders with stratified sampling.
    """

    # Define transforms
    base_transform = Compose([
        Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        ToTensor()
    ])
    
    if augmentation:
        train_transform = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=15),
            Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            ToTensor()
        ])
    else:
        train_transform = base_transform

    # Load the same dataset twice with different transforms
    train_full_dataset = ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_full_dataset = ImageFolder(f"{data_dir}/train", transform=base_transform)

    # Get labels for stratified split
    labels = [label for _, label in train_full_dataset.samples]

    # Stratified split based on class labels
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, val_indices = next(sss.split(np.zeros(len(labels)), labels))

    # Create train and validation subsets from the datasets with appropriate transforms
    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_loader, val_loader


def get_testing_dataloader(data_dir, batch_size=32, num_workers=4):
    """
    Returns test DataLoader.
    """
    transform = Compose([Resize((256, 256),interpolation=InterpolationMode.BILINEAR), ToTensor()])
    full_dataset = ImageFolder(f"{data_dir}/val", transform=transform)

    test_dataloader = DataLoader(full_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    
    return test_dataloader