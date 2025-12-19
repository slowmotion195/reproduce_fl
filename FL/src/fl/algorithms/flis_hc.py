import os
import json
import copy
import numpy as np
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from src.data.datasets import get_torchvision_dataset
from src.data.transforms import build_transforms
from src.data.partition import partition_labeldir, count_by_label
from src.data.cache import SplitCache, SharedCache
from src.data.loaders import build_global_loaders, build_client_loaders, build_local_view_test_split
from src.data.wrapper import DatasetSplit

from src.models.factory import build_model
from src.fl.client import Client
from src.fl.eval import eval_model
from src.fl.algorithms.fedavg import fedavg_state_dicts

from src.clustering.similarity import compute_pred_matrix_and_sim
from src.clustering.form_clusters import form_clusters


def _save_config(run_dir: str, cfg: Dict[str, Any]):
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def _build_shared_indices(y, nsamples_shared: int, nclasses: int, seed: int):
    """
    Balanced by class: take N = nsamples_shared//nclasses per class.
    """
    rng = np.random.default_rng(seed)
    per = nsamples_shared // nclasses
    idxs = []
    for k in range(nclasses):
        idx_k = np.where(y == k)[0]
        if len(idx_k) < per:
            raise ValueError(f"class {k} has {len(idx_k)} < per_class {per} in shared source")
        chosen = rng.choice(idx_k, size=per, replace=False)
        idxs.extend(chosen.tolist())
    rng.shuffle(idxs)
    return np.array(idxs, dtype=np.int64)


def run_flis_hc(cfg, device, logger, metrics, run_dir: str):
    _save_config(run_dir, cfg)

    dataset = cfg["dataset"]
    data_root = cfg["data_root"]
    seed = int(cfg["seed"])

    # 1) dataset
    train_tf, test_tf = build_transforms(dataset)
    train_ds = get_torchvision_dataset(dataset, data_root, train=True, transform=train_tf, download=True)
    test_ds = get_torchvision_dataset(dataset, data_root, train=False, transform=test_tf, download=True)

    y_train = np.array(train_ds.targets)
    y_test = np.array(test_ds.targets)

    # 2) partition (with cache)
    split_cache = SplitCache(data_root, dataset)
    split_key = split_cache.key(cfg["partition"], float(cfg["beta"]), int(cfg["num_users"]), seed, bool(cfg["local_view"]))
    cached = split_cache.load(split_key)

    if cached is None:
        logger.info("No cached split found. Creating partition...")
        if cfg["partition"] != "noniid-labeldir":
            raise ValueError("This skeleton currently supports partition=noniid-labeldir only.")
        net_dataidx_map = partition_labeldir(y_train, int(cfg["num_users"]), float(cfg["beta"]), seed=seed)

        train_stats = {cid: count_by_label(y_train, idxs) for cid, idxs in net_dataidx_map.items()}
        if bool(cfg["local_view"]):
            net_dataidx_map_test = build_local_view_test_split(y_test, train_stats)
        else:
            net_dataidx_map_test = None

        cached = {
            "net_dataidx_map": net_dataidx_map,
            "net_dataidx_map_test": net_dataidx_map_test,
            "train_stats": train_stats,
        }
        split_cache.save(split_key, cached)
        logger.info(f"Saved split cache: {split_key}")
    else:
        logger.info(f"Loaded split cache: {split_key}")

    net_dataidx_map = {int(k): v for k, v in cached["net_dataidx_map"].items()}
    net_dataidx_map_test = cached["net_dataidx_map_test"]
    if net_dataidx_map_test is not None:
        net_dataidx_map_test = {int(k): v for k, v in net_dataidx_map_test.items()}
    train_stats = {int(k): v for k, v in cached["train_stats"].items()}

    logger.info(f"Train stats example (client0): {train_stats.get(0)}")

    # 3) dataloaders
    train_dl_global, test_dl_global = build_global_loaders(
        train_ds, test_ds,
        batch_size=int(cfg.get("batch_size", 128)),
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
    )

    client_train, client_test = build_client_loaders(
        train_ds, test_ds,
        net_dataidx_map=net_dataidx_map,
        net_dataidx_map_test=net_dataidx_map_test,
        local_bs=int(cfg["local_bs"]),
        test_bs=int(cfg.get("test_bs", 256)),
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
    )

    # 4) shared loader (cached)
    shared_cache = SharedCache(data_root, dataset)
    shared_key = shared_cache.key(int(cfg["nsamples_shared"]), int(cfg["num_classes"]), seed, cfg["shared_source"])
    shared_idxs = shared_cache.load(shared_key)

    if shared_idxs is None:
        logger.info("No cached shared idx found. Building shared set...")
        if cfg["shared_source"] == "test":
            shared_idxs = _build_shared_indices(y_test, int(cfg["nsamples_shared"]), int(cfg["num_classes"]), seed)
            shared_ds = DatasetSplit(test_ds, shared_idxs)
        else:
            shared_idxs = _build_shared_indices(y_train, int(cfg["nsamples_shared"]), int(cfg["num_classes"]), seed)
            shared_ds = DatasetSplit(train_ds, shared_idxs)

        shared_cache.save(shared_key, shared_idxs)
        logger.info(f"Saved shared cache: {shared_key}")
    else:
        logger.info(f"Loaded shared cache: {shared_key}")
        if cfg["shared_source"] == "test":
            shared_ds = DatasetSplit(test_ds, shared_idxs)
        else:
            shared_ds = DatasetSplit(train_ds, shared_idxs)

    shared_loader = DataLoader(
        shared_ds,
        batch_size=int(cfg["nsamples_shared"]) // int(cfg["num_classes"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
    )

    # 5) init global + clients
    base_model = build_model(cfg["model"], int(cfg["in_ch"]), int(cfg["num_classes"]))
    base_model.to(device)
    server_state = copy.deepcopy(base_model.state_dict())

    clients = {}
    for cid in range(int(cfg["num_users"])):
        local_model = build_model(cfg["model"], int(cfg["in_ch"]), int(cfg["num_classes"]))
        local_model.load_state_dict(server_state, strict=True)
        clients[cid] = Client(
            cid=cid,
            model=local_model,
            train_loader=client_train[cid],
            test_loader=client_test[cid],
            lr=float(cfg["lr"]),
            momentum=float(cfg["momentum"]),
            weight_decay=float(cfg["weight_decay"]),
            device=device
        )

    # 6) pre-federation: 1 local train then compute similarity & clusters
    logger.info("Pre-federation: local warmup training 1 pass for clustering...")
    for cid in range(int(cfg["num_users"])):
        clients[cid].set_state_dict(copy.deepcopy(server_state))
        _ = clients[cid].train_local(local_ep=1)

    client_ids = np.arange(int(cfg["num_users"]))
    sim_mat = compute_pred_matrix_and_sim(
        clients=clients,
        client_ids=list(client_ids),
        shared_loader=shared_loader,
        device=device,
        nclasses=int(cfg["num_classes"])
    )
    clusters = form_clusters(sim_mat, client_ids, alpha=float(cfg["cluster_alpha"]))

    np.save(os.path.join(run_dir, "sim_mat.npy"), sim_mat)
    with open(os.path.join(run_dir, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in clusters.items()}, f, ensure_ascii=False, indent=2)

    # reset to server
    for cid in range(int(cfg["num_users"])):
        clients[cid].set_state_dict(copy.deepcopy(server_state))

    # 7) federation
    best_glob_acc = -1.0
    best_glob_state = None

    for rnd in range(int(cfg["rounds"])):
        m = max(int(float(cfg["frac"]) * int(cfg["num_users"])), 1)
        selected = np.random.choice(int(cfg["num_users"]), m, replace=False)

        # local steps
        local_states = []
        local_weights = []
        local_losses = []
        init_accs, final_accs = [], []

        for cid in selected:
            # cluster-based init after round1
            if rnd >= 1:
                members = clusters[int(np.where(client_ids == cid)[0][0])]  # cluster list for this index
                # FedAvg within cluster members
                sd_list = [clients[mid].get_state_dict() for mid in members]
                total = sum(len(net_dataidx_map[mid]) for mid in members)
                freqs = [len(net_dataidx_map[mid]) / total for mid in members]
                w_cluster = fedavg_state_dicts(sd_list, freqs, device=device)
                clients[cid].set_state_dict(w_cluster)

            # eval before
            loss0, acc0 = clients[cid].eval_test()
            init_accs.append(acc0)

            # train
            loss_tr = clients[cid].train_local(local_ep=int(cfg["local_ep"]))
            local_losses.append(loss_tr)

            # eval after
            loss1, acc1 = clients[cid].eval_test()
            final_accs.append(acc1)

            # collect for global avg
            local_states.append(clients[cid].get_state_dict())
            local_weights.append(len(net_dataidx_map[cid]))

            # update client best
            if acc1 > clients[cid].best_acc:
                clients[cid].best_acc = acc1

        # global FedAvg
        total_w = sum(local_weights)
        freqs = [w / total_w for w in local_weights]
        w_glob = fedavg_state_dicts(local_states, freqs, device=device)

        base_model.load_state_dict(w_glob, strict=True)
        _, glob_acc = eval_model(base_model, test_dl_global, device)

        if glob_acc > best_glob_acc:
            best_glob_acc = glob_acc
            best_glob_state = copy.deepcopy(w_glob)
            if bool(cfg["save_best"]):
                torch.save(best_glob_state, os.path.join(run_dir, "checkpoints", "global_best.pt"))

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
                f"[Round {rnd+1:03d}] "
                f"loss={rec['avg_train_loss']:.4f} "
                f"init_acc={rec['avg_init_acc']:.2f} "
                f"final_acc={rec['avg_final_acc']:.2f} "
                f"glob_acc={rec['global_acc']:.2f} "
                f"best={best_glob_acc:.2f}"
            )

    # 8) final stats
    test_accs = []
    train_accs = []
    for cid in range(int(cfg["num_users"])):
        _, a_te = clients[cid].eval_test()
        _, a_tr = clients[cid].eval_train()
        test_accs.append(a_te)
        train_accs.append(a_tr)

    logger.info(f"Final avg client train acc: {float(np.mean(train_accs)):.2f}")
    logger.info(f"Final avg client test  acc: {float(np.mean(test_accs)):.2f}")
    logger.info(f"Best global acc: {best_glob_acc:.2f}")
