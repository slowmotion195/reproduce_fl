import numpy as np
from typing import Dict, List, Tuple

def partition_labeldir(
    y: np.ndarray,
    num_users: int,
    beta: float,
    seed: int,
    min_require_size: int = 10,
) -> Dict[int, List[int]]:
    """
    Dirichlet label distribution partition (classic noniid-labeldir).

    Returns: net_dataidx_map: {client_id: [sample_indices]}
    """
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    K = int(np.max(y) + 1)

    idx_by_class = [np.where(y == k)[0] for k in range(K)]
    for k in range(K):
        rng.shuffle(idx_by_class[k])

    net_dataidx_map = {i: [] for i in range(num_users)}
    min_size = 0

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = idx_by_class[k]
            proportions = rng.dirichlet(np.repeat(beta, num_users))
            # balance (optional): avoid one client taking too much when already large
            proportions = np.array([p * (len(idx_batch[j]) < n / num_users) for j, p in enumerate(proportions)])
            proportions = proportions / proportions.sum()
            cuts = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            splits = np.split(idx_k, cuts)
            for j in range(num_users):
                idx_batch[j].extend(splits[j].tolist())

        min_size = min(len(b) for b in idx_batch)

    for j in range(num_users):
        rng.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map

def count_by_label(y: np.ndarray, idxs: List[int]) -> Dict[int, int]:
    labels, cnts = np.unique(y[idxs], return_counts=True)
    return {int(l): int(c) for l, c in zip(labels, cnts)}
