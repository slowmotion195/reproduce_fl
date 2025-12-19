import os
import json
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_noniid_heatmap(
    train_stats: Dict[int, Dict[int, int]],
    num_users: int,
    num_classes: int,
    save_path: str,
    title: str = "Non-IID data allocation heatmap (train)"
):
    """
    train_stats: {cid: {label: count}}
    Heatmap matrix M[cid, label] = count
    """
    M = np.zeros((num_users, num_classes), dtype=np.float32)
    for cid in range(num_users):
        stat = train_stats.get(cid, {})
        for lab, cnt in stat.items():
            if 0 <= int(lab) < num_classes:
                M[cid, int(lab)] = float(cnt)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure()
    plt.imshow(M, aspect="auto")
    plt.colorbar()
    plt.xlabel("Class label")
    plt.ylabel("Client ID")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plot_compare_curves(
    metrics_items: List[Tuple[str, str]],
    save_acc_path: str,
    save_loss_path: str,
    acc_key: str = "global_acc",
    loss_key: str = "avg_train_loss"
):
    """
    metrics_items: list of (label, metrics_jsonl_path)
    """
    os.makedirs(os.path.dirname(save_acc_path), exist_ok=True)

    # Accuracy plot
    plt.figure()
    for label, path in metrics_items:
        rows = _read_jsonl(path)
        xs = [r.get("round") for r in rows]
        ys = [r.get(acc_key) for r in rows]
        plt.plot(xs, ys, label=label)
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Global accuracy comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_acc_path, dpi=200)
    plt.close()

    # Loss plot
    plt.figure()
    for label, path in metrics_items:
        rows = _read_jsonl(path)
        xs = [r.get("round") for r in rows]
        ys = [r.get(loss_key) for r in rows]
        plt.plot(xs, ys, label=label)
    plt.xlabel("Round")
    plt.ylabel("Average train loss")
    plt.title("Train loss comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_loss_path, dpi=200)
    plt.close()
