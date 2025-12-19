import torch
import torch.nn.functional as F

@torch.no_grad()
def eval_model(model, loader, device):
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        total_loss += F.cross_entropy(out, y, reduction="sum").item()
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / max(1, total), 100.0 * correct / max(1, total)
