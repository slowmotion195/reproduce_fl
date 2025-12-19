import copy
import torch
from typing import List, Optional

def fedavg_state_dicts(state_dicts: List[dict], weights: Optional[List[float]] = None, device=None) -> dict:
    """
    Device-robust FedAvg: never hardcode .cuda()
    """
    assert len(state_dicts) > 0
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    assert len(weights) == len(state_dicts)

    out = copy.deepcopy(state_dicts[0])
    for k in out.keys():
        out[k] = out[k].to(device) * weights[0] if device is not None else out[k] * weights[0]

    for i in range(1, len(state_dicts)):
        sd = state_dicts[i]
        for k in out.keys():
            tk = sd[k].to(device) if device is not None else sd[k]
            out[k] = out[k] + tk * weights[i]

    return out
