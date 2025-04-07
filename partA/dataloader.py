import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode, RandomHorizontalFlip, RandomRotation


def get_training_dataloaders(data_dir, batch_size=32, num_workers=4, augmentaion=False):
    """
    Returns train and validation DataLoaders with stratified sampling.
    """
    transform = Compose([Resize((256, 256),interpolation=InterpolationMode.BILINEAR), ToTensor()])
    full_dataset = ImageFolder(f"{data_dir}/train", transform=transform)

    # Stratified split based on class labels
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers)

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