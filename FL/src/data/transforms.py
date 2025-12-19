import torch
import torchvision.transforms as T

def build_transforms(dataset: str):
    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        test_tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        return train_tf, test_tf

    raise ValueError(f"Unsupported dataset: {dataset}")
