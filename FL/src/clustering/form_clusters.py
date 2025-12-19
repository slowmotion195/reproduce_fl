import numpy as np
import copy
from typing import Dict, List

def form_clusters(sim_mat: np.ndarray, client_ids: np.ndarray, alpha: float) -> Dict[int, List[int]]:
    """
    Your original logic: for each client i, create a cluster list of ids with sim >= alpha
    plus ultra-high sim >= 0.96.
    Returns dict: clusters[i_index] -> list of client ids
    """
    nclients = sim_mat.shape[0]
    clusters = {i: None for i in range(nclients)}

    for i in range(nclients):
        temp = np.vstack([np.arange(nclients), sim_mat[i]])
        temp = temp[:, temp[1, :].argsort()[::-1]]
        sorted_idx = temp[0]
        sorted_sim = temp[1]

        cc = [int(client_ids[i])]
        index = 0
        while index < nclients:
            simv = sorted_sim[index]
            j = int(sorted_idx[index])
            if simv >= 0.96:
                if i != j:
                    cc.append(int(client_ids[j]))
                index += 1
            elif simv >= alpha:
                cc.append(int(client_ids[j]))
                index += 1
            else:
                break

        clusters[i] = copy.deepcopy(cc)

    return clusters
