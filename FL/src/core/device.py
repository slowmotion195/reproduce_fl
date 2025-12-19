import torch

def build_device(gpu: int):
    if gpu < 0:
        return torch.device("cpu")
    if torch.cuda.is_available():
        # single GPU assumed
        torch.cuda.set_device(gpu)
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")
