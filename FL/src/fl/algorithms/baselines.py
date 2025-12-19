import os
import copy
import numpy as np
from typing import Dict, Any, List, Optional

import torch

from src.fl.algorithms.fedavg import fedavg_state_dicts
from src.fl.eval import eval_model


def build_random_clusters(num_users: int, cluster_size: int, seed: int) -> Dict[int, List[int]]:
    """
    Return clusters_by_cid: {cid: [members...]} where key is cid.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_users).tolist()
    clusters = []
    for i in range(0, num_users, cluster_size):
        clusters.append(perm[i:i + cluster_size])

    clusters_by_cid = {}
    for group in clusters:
        for cid in group:
            clusters_by_cid[cid] = list(group)
    return clusters_by_cid


def run_fedavg(
    cfg: Dict[str, Any],
    device,
    logger,
    metrics,
    run_dir: str,
    base_model,
    clients: Dict[int, object],
    net_dataidx_map: Dict[int, List[int]],
    test_dl_global,
    seed_offset: int = 0,
    save_best_name: str = "global_best_fedavg.pt",
):
    """
    Baseline 1: vanilla FedAvg (no clustering).
    """
    best_glob_acc = -1.0
    best_glob_state = None

    rng = np.random.default_rng(int(cfg["seed"]) + seed_offset)

    for rnd in range(int(cfg["rounds"])):
        m = max(int(float(cfg["frac"]) * int(cfg["num_users"])), 1)
        selected = rng.choice(int(cfg["num_users"]), m, replace=False)

        local_states = []
        local_weights = []
        local_losses = []
        init_accs, final_accs = [], []

        for cid in selected:
            loss0, acc0 = clients[cid].eval_test()
            init_accs.append(acc0)

            loss_tr = clients[cid].train_local(local_ep=int(cfg["local_ep"]))
            local_losses.append(loss_tr)

            loss1, acc1 = clients[cid].eval_test()
            final_accs.append(acc1)

            local_states.append(clients[cid].get_state_dict())
            local_weights.append(len(net_dataidx_map[cid]))

        total_w = sum(local_weights)
        freqs = [w / total_w for w in local_weights]
        w_glob = fedavg_state_dicts(local_states, freqs, device=device)

        base_model.load_state_dict(w_glob, strict=True)
        _, glob_acc = eval_model(base_model, test_dl_global, device)

        if glob_acc > best_glob_acc:
            best_glob_acc = glob_acc
            best_glob_state = copy.deepcopy(w_glob)
            if bool(cfg.get("save_best", True)):
                torch.save(best_glob_state, os.path.join(run_dir, "checkpoints", save_best_name))

        rec = {
            "round": rnd + 1,
            "selected": selected.tolist(),
            "avg_train_loss": float(np.mean(local_losses)) if len(local_losses) else None,
            "avg_init_acc": float(np.mean(init_accs)) if len(init_accs) else None,
            "avg_final_acc": float(np.mean(final_accs)) if len(final_accs) else None,
            "global_acc": float(glob_acc),
            "best_global_acc": float(best_glob_acc),
        }
        metrics.write(rec)

        if (rnd + 1) % int(cfg["print_freq"]) == 0 or rnd == 0:
            logger.info(
                f"[FedAvg Round {rnd+1:03d}] "
                f"loss={rec['avg_train_loss']:.4f} "
                f"init_acc={rec['avg_init_acc']:.2f} "
                f"final_acc={rec['avg_final_acc']:.2f} "
                f"glob_acc={rec['global_acc']:.2f} "
                f"best={best_glob_acc:.2f}"
            )


def run_random_cluster(
    cfg: Dict[str, Any],
    device,
    logger,
    metrics,
    run_dir: str,
    base_model,
    clients: Dict[int, object],
    net_dataidx_map: Dict[int, List[int]],
    test_dl_global,
    clusters_by_cid: Dict[int, List[int]],
    seed_offset: int = 10,
    save_best_name: str = "global_best_randcluster.pt",
):
    """
    Baseline 2: random clustering + cluster-init (like your FLIS round>=1 init style).
    """
    best_glob_acc = -1.0
    best_glob_state = None

    rng = np.random.default_rng(int(cfg["seed"]) + seed_offset)

    for rnd in range(int(cfg["rounds"])):
        m = max(int(float(cfg["frac"]) * int(cfg["num_users"])), 1)
        selected = rng.choice(int(cfg["num_users"]), m, replace=False)

        local_states = []
        local_weights = []
        local_losses = []
        init_accs, final_accs = [], []

        for cid in selected:
            # cluster init after round 1
            if rnd >= 1:
                members = clusters_by_cid[cid]
                sd_list = [clients[mid].get_state_dict() for mid in members]
                total = sum(len(net_dataidx_map[mid]) for mid in members)
                freqs = [len(net_dataidx_map[mid]) / total for mid in members]
                w_cluster = fedavg_state_dicts(sd_list, freqs, device=device)
                clients[cid].set_state_dict(w_cluster)

            loss0, acc0 = clients[cid].eval_test()
            init_accs.append(acc0)

            loss_tr = clients[cid].train_local(local_ep=int(cfg["local_ep"]))
            local_losses.append(loss_tr)

            loss1, acc1 = clients[cid].eval_test()
            final_accs.append(acc1)

            local_states.append(clients[cid].get_state_dict())
            local_weights.append(len(net_dataidx_map[cid]))

        total_w = sum(local_weights)
        freqs = [w / total_w for w in local_weights]
        w_glob = fedavg_state_dicts(local_states, freqs, device=device)

        base_model.load_state_dict(w_glob, strict=True)
        _, glob_acc = eval_model(base_model, test_dl_global, device)

        if glob_acc > best_glob_acc:
            best_glob_acc = glob_acc
            best_glob_state = copy.deepcopy(w_glob)
            if bool(cfg.get("save_best", True)):
                torch.save(best_glob_state, os.path.join(run_dir, "checkpoints", save_best_name))

        rec = {
            "round": rnd + 1,
            "selected": selected.tolist(),
            "avg_train_loss": float(np.mean(local_losses)) if len(local_losses) else None,
            "avg_init_acc": float(np.mean(init_accs)) if len(init_accs) else None,
            "avg_final_acc": float(np.mean(final_accs)) if len(final_accs) else None,
            "global_acc": float(glob_acc),
            "best_global_acc": float(best_glob_acc),
        }
        metrics.write(rec)

        if (rnd + 1) % int(cfg["print_freq"]) == 0 or rnd == 0:
            logger.info(
                f"[RandCluster Round {rnd+1:03d}] "
                f"loss={rec['avg_train_loss']:.4f} "
                f"init_acc={rec['avg_init_acc']:.2f} "
                f"final_acc={rec['avg_final_acc']:.2f} "
                f"glob_acc={rec['global_acc']:.2f} "
                f"best={best_glob_acc:.2f}"
            )
