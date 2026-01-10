# code/data.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import CIFAR10_MEAN, CIFAR10_STD, BATCH_SIZE, DATA_DIR
from utils import ensure_dir


def get_transforms_normalized():
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_t, test_t


def get_transforms_pixelspace():
    # come nel tuo snippet: NO Normalize, output in [0,1]
    train_t = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_t = transforms.Compose([
        transforms.ToTensor(),
    ])
    return train_t, test_t


def get_cifar10_loaders(batch_size: int = BATCH_SIZE, device: torch.device | None = None):
    """
    Loader "standard" per training/eval: immagini NORMALIZZATE (come nel training).
    """
    ensure_dir(DATA_DIR)
    train_t, test_t = get_transforms_normalized()

    train_set = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=train_t)
    test_set = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=test_t)

    num_workers = 2 if os.name != "nt" else 0
    pin_memory = (device is not None and device.type == "cuda")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


def get_cifar10_loaders_pixelspace(batch_size: int = BATCH_SIZE, device: torch.device | None = None):
    """
    Loader per UAP: immagini in pixel-space [0,1] (NO Normalize).
    """
    ensure_dir(DATA_DIR)
    train_t, test_t = get_transforms_pixelspace()

    train_set = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=train_t)
    test_set = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False, transform=test_t)

    num_workers = 2 if os.name != "nt" else 0
    pin_memory = (device is not None and device.type == "cuda")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


def normalize_batch(x_pix: torch.Tensor,
                    mean=CIFAR10_MEAN,
                    std=CIFAR10_STD) -> torch.Tensor:
    """
    x_pix: [B,3,32,32] in [0,1]
    ritorna: (x - mean)/std
    """
    mean_t = torch.tensor(mean, device=x_pix.device, dtype=x_pix.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x_pix.device, dtype=x_pix.dtype).view(1, 3, 1, 1)
    return (x_pix - mean_t) / std_t
