import os
import json
import numpy as np
from typing import Dict, Any, List, Optional

class SplitCache:
    def __init__(self, data_root: str, dataset: str):
        self.base = os.path.join(data_root, "processed", dataset, "splits")
        os.makedirs(self.base, exist_ok=True)

    def key(self, partition: str, beta: float, num_users: int, seed: int, local_view: bool) -> str:
        lv = "localview" if local_view else "globaltest"
        return f"{partition}_beta{beta}_users{num_users}_seed{seed}_{lv}.json"

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.base, key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj

    def save(self, key: str, obj: Dict[str, Any]):
        path = os.path.join(self.base, key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

class SharedCache:
    def __init__(self, data_root: str, dataset: str):
        self.base = os.path.join(data_root, "processed", dataset, "shared")
        os.makedirs(self.base, exist_ok=True)

    def key(self, nsamples_shared: int, nclasses: int, seed: int, source: str) -> str:
        return f"shared_{source}_ns{nsamples_shared}_c{nclasses}_seed{seed}.npy"

    def load(self, key: str):
        path = os.path.join(self.base, key)
        if not os.path.exists(path):
            return None
        return np.load(path)

    def save(self, key: str, idxs: np.ndarray):
        path = os.path.join(self.base, key)
        np.save(path, idxs)
