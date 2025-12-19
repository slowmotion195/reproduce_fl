import os
from torchvision import datasets

def get_torchvision_dataset(dataset: str, data_root: str, train: bool, transform, download: bool = True):
    raw_root = os.path.join(data_root, "raw")
    os.makedirs(raw_root, exist_ok=True)

    if dataset == "cifar10":
        return datasets.CIFAR10(root=raw_root, train=train, transform=transform, download=download)
    if dataset == "cifar100":
        return datasets.CIFAR100(root=raw_root, train=train, transform=transform, download=download)

    raise ValueError(f"Unsupported dataset: {dataset}")
