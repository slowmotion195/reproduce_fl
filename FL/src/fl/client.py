import torch
import torch.nn as nn
import torch.nn.functional as F

class Client:
    def __init__(self, cid: int, model: torch.nn.Module, train_loader, test_loader,
                 lr: float, momentum: float, weight_decay: float, device):
        self.cid = cid
        self.net = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        self.best_acc = 0.0

    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict, strict=True)

    def get_state_dict(self):
        return self.net.state_dict()

    def get_net(self):
        return self.net

    def train_local(self, local_ep: int):
        self.net.to(self.device)
        self.net.train()

        epoch_losses = []
        for _ in range(local_ep):
            batch_losses = []
            for x, y in self.train_loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                self.opt.zero_grad(set_to_none=True)
                out = self.net(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.opt.step()
                batch_losses.append(loss.item())
            epoch_losses.append(sum(batch_losses) / max(1, len(batch_losses)))
        return sum(epoch_losses) / max(1, len(epoch_losses))

    @torch.no_grad()
    def eval_loader(self, loader):
        self.net.to(self.device)
        self.net.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            out = self.net(x)
            total_loss += F.cross_entropy(out, y, reduction="sum").item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        return total_loss / max(1, total), 100.0 * correct / max(1, total)

    def eval_test(self):
        return self.eval_loader(self.test_loader)

    def eval_train(self):
        return self.eval_loader(self.train_loader)
