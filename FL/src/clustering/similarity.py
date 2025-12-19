import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

@torch.no_grad()
def compute_pred_matrix_and_sim(clients: Dict[int, object], client_ids: List[int], shared_loader, device, nclasses: int):
    """
    Compute per-client prediction one-hot matrix on shared set, then cosine similarity.

    Returns:
      sim_mat: [nclients, nclients] numpy
    """
    # collect predictions for each client across shared_loader
    # store as list of tensors then concat -> [N, C] one-hot
    pred_onehot = {}

    for cid in client_ids:
        model = clients[cid].get_net()
        model.to(device)
        model.eval()
        preds = []
        for x, _ in shared_loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            p = out.argmax(dim=1)
            preds.append(F.one_hot(p, num_classes=nclasses).float().cpu())
        pred_onehot[cid] = torch.cat(preds, dim=0)  # [N, C]

    # compute cosine similarity via Frobenius inner product
    n = len(client_ids)
    sim_mat = np.zeros((n, n), dtype=np.float32)

    mats = [pred_onehot[cid].numpy() for cid in client_ids]
    norms = [np.linalg.norm(m, ord="fro") + 1e-12 for m in mats]

    for i in range(n):
        for j in range(n):
            sim_mat[i, j] = float((mats[i] * mats[j]).sum() / (norms[i] * norms[j]))
    return sim_mat
