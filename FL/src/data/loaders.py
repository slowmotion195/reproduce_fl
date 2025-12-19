import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader

from .wrapper import DatasetSplit
from .partition import count_by_label

def build_global_loaders(train_ds, test_ds, batch_size: int, num_workers: int, pin_memory: bool):
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_dl, test_dl

def build_client_loaders(
    train_ds,
    test_ds,
    net_dataidx_map: Dict[int, List[int]],
    net_dataidx_map_test: Optional[Dict[int, List[int]]],
    local_bs: int,
    test_bs: int,
    num_workers: int,
    pin_memory: bool
):
    client_train = {}
    client_test = {}
    for cid, idxs in net_dataidx_map.items():
        tr = DataLoader(
            DatasetSplit(train_ds, idxs),
            batch_size=local_bs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        if net_dataidx_map_test is None:
            te = DataLoader(test_ds, batch_size=test_bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        else:
            te = DataLoader(
                DatasetSplit(test_ds, net_dataidx_map_test[cid]),
                batch_size=test_bs,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        client_train[cid] = tr
        client_test[cid] = te
    return client_train, client_test

def build_local_view_test_split(y_test: np.ndarray, train_stats: Dict[int, Dict[int, int]]):
    """
    local_view=True: each client gets test samples only for labels it owns.
    train_stats: {cid: {label: count}}
    """
    net_dataidx_map_test = {cid: [] for cid in train_stats.keys()}
    for cid, stat in train_stats.items():
        labels = list(stat.keys())
        for l in labels:
            idx_l = np.where(y_test == l)[0]
            net_dataidx_map_test[cid].extend(idx_l.tolist())
    return net_dataidx_map_test
