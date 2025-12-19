import os
import json
import copy
import numpy as np
from typing import Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader

from src.core.metrics import MetricsWriter

from src.data.datasets import get_torchvision_dataset
from src.data.transforms import build_transforms
from src.data.partition import partition_labeldir, count_by_label
from src.data.cache import SplitCache, SharedCache
from src.data.loaders import build_global_loaders, build_client_loaders, build_local_view_test_split
from src.data.wrapper import DatasetSplit

from src.models.factory import build_model
from src.fl.client import Client

from src.clustering.similarity import compute_pred_matrix_and_sim
from src.clustering.form_clusters import form_clusters

from src.vis.plots import plot_noniid_heatmap, plot_compare_curves
from src.fl.algorithms.baselines import run_fedavg, run_random_cluster, build_random_clusters


def _ensure_subrun_dir(parent_run_dir: str, name: str) -> str:
    sub = os.path.join(parent_run_dir, name)
    os.makedirs(sub, exist_ok=True)
    os.makedirs(os.path.join(sub, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(sub, "artifacts"), exist_ok=True)
    return sub


def _save_config(run_dir: str, cfg: Dict[str, Any], filename: str = "config.json"):
    with open(os.path.join(run_dir, filename), "w", encoding="utf-8") as f:
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


def _prepare_data(cfg: Dict[str, Any], logger) -> Tuple:
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
            raise ValueError("compare currently supports partition=noniid-labeldir only.")
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

    return (
        train_ds, test_ds, y_train, y_test,
        net_dataidx_map, net_dataidx_map_test, train_stats,
        train_dl_global, test_dl_global,
        client_train, client_test,
        shared_loader
    )


def _init_models_and_clients(cfg: Dict[str, Any], device, client_train, client_test):
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
    return base_model, server_state, clients


def _precluster_flis_hc(cfg: Dict[str, Any], device, logger, server_state, clients, shared_loader, run_dir: str):
    """
    Exactly the same pre-federation idea as your flis_hc.py:
    1) each client warmup train 1 epoch
    2) compute sim_mat on shared
    3) form_clusters
    """
    logger.info("FLIS(HC) Pre-federation: local warmup training 1 pass for clustering...")
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

    # save artifacts
    np.save(os.path.join(run_dir, "sim_mat.npy"), sim_mat)
    with open(os.path.join(run_dir, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in clusters.items()}, f, ensure_ascii=False, indent=2)

    # reset to server
    for cid in range(int(cfg["num_users"])):
        clients[cid].set_state_dict(copy.deepcopy(server_state))

    return clusters


def run_compare(cfg: Dict[str, Any], device, logger, run_dir: str):
    """
    Run three experiments in one run_dir:
      1) FLIS(HC)
      2) FedAvg (no cluster)
      3) Random Cluster + cluster-init
    Save:
      - artifacts/noniid_heatmap.png (once)
      - artifacts/compare_acc.png
      - artifacts/compare_loss.png
      - metrics_flis.jsonl / metrics_fedavg.jsonl / metrics_randcluster.jsonl
    """
    artifacts_dir = os.path.join(run_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    _save_config(run_dir, cfg, filename="config_compare.json")

    # Prepare data once (fair comparison)
    (
        train_ds, test_ds, y_train, y_test,
        net_dataidx_map, net_dataidx_map_test, train_stats,
        train_dl_global, test_dl_global,
        client_train, client_test,
        shared_loader
    ) = _prepare_data(cfg, logger)

    # (A) plot Non-IID heatmap once
    heatmap_path = os.path.join(artifacts_dir, "noniid_heatmap.png")
    plot_noniid_heatmap(
        train_stats=train_stats,
        num_users=int(cfg["num_users"]),
        num_classes=int(cfg["num_classes"]),
        save_path=heatmap_path
    )
    logger.info(f"Saved Non-IID heatmap to: {heatmap_path}")

    # Prepare sub run dirs (keep outputs separated, do not overwrite checkpoints)
    flis_dir = _ensure_subrun_dir(run_dir, "exp_flis_hc")
    fedavg_dir = _ensure_subrun_dir(run_dir, "exp_fedavg")
    rand_dir = _ensure_subrun_dir(run_dir, "exp_randcluster")

    # ===== 1) FLIS(HC) =====
    logger.info("=== Running experiment 1/3: FLIS(HC) ===")
    _save_config(flis_dir, cfg, filename="config.json")

    base_model, server_state, clients = _init_models_and_clients(cfg, device, client_train, client_test)
    clusters = _precluster_flis_hc(cfg, device, logger, server_state, clients, shared_loader, run_dir=flis_dir)

    # metrics writer (custom filename in subdir)
    metrics_flis = MetricsWriter(flis_dir, filename="metrics_flis.jsonl")

    # Use your original FLIS(HC) training loop logic by inlining the same loop
    # (kept consistent with your flis_hc.py; no extra structural changes)
    best_glob_acc = -1.0
    best_glob_state = None
    client_ids = np.arange(int(cfg["num_users"]))

    for rnd in range(int(cfg["rounds"])):
        m = max(int(float(cfg["frac"]) * int(cfg["num_users"])), 1)
        selected = np.random.choice(int(cfg["num_users"]), m, replace=False)

        local_states = []
        local_weights = []
        local_losses = []
        init_accs, final_accs = [], []

        for cid in selected:
            if rnd >= 1:
                members = clusters[int(np.where(client_ids == cid)[0][0])]
                sd_list = [clients[mid].get_state_dict() for mid in members]
                total = sum(len(net_dataidx_map[mid]) for mid in members)
                freqs = [len(net_dataidx_map[mid]) / total for mid in members]
                # reuse device-robust fedavg
                from src.fl.algorithms.fedavg import fedavg_state_dicts
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

            if acc1 > clients[cid].best_acc:
                clients[cid].best_acc = acc1

        total_w = sum(local_weights)
        freqs = [w / total_w for w in local_weights]
        from src.fl.algorithms.fedavg import fedavg_state_dicts
        w_glob = fedavg_state_dicts(local_states, freqs, device=device)

        base_model.load_state_dict(w_glob, strict=True)
        from src.fl.eval import eval_model
        _, glob_acc = eval_model(base_model, test_dl_global, device)

        if glob_acc > best_glob_acc:
            best_glob_acc = glob_acc
            best_glob_state = copy.deepcopy(w_glob)
            if bool(cfg.get("save_best", True)):
                torch.save(best_glob_state, os.path.join(flis_dir, "checkpoints", "global_best.pt"))

        rec = {
            "round": rnd + 1,
            "selected": selected.tolist(),
            "avg_train_loss": float(np.mean(local_losses)) if len(local_losses) else None,
            "avg_init_acc": float(np.mean(init_accs)) if len(init_accs) else None,
            "avg_final_acc": float(np.mean(final_accs)) if len(final_accs) else None,
            "global_acc": float(glob_acc),
            "best_global_acc": float(best_glob_acc),
        }
        metrics_flis.write(rec)

        if (rnd + 1) % int(cfg["print_freq"]) == 0 or rnd == 0:
            logger.info(
                f"[FLIS(HC) Round {rnd+1:03d}] "
                f"loss={rec['avg_train_loss']:.4f} "
                f"init_acc={rec['avg_init_acc']:.2f} "
                f"final_acc={rec['avg_final_acc']:.2f} "
                f"glob_acc={rec['global_acc']:.2f} "
                f"best={best_glob_acc:.2f}"
            )

    metrics_flis.close()

    # ===== 2) FedAvg =====
    logger.info("=== Running experiment 2/3: FedAvg (no cluster) ===")
    _save_config(fedavg_dir, cfg, filename="config.json")

    base_model2, server_state2, clients2 = _init_models_and_clients(cfg, device, client_train, client_test)
    metrics_fed = MetricsWriter(fedavg_dir, filename="metrics_fedavg.jsonl")
    run_fedavg(
        cfg=cfg,
        device=device,
        logger=logger,
        metrics=metrics_fed,
        run_dir=fedavg_dir,
        base_model=base_model2,
        clients=clients2,
        net_dataidx_map=net_dataidx_map,
        test_dl_global=test_dl_global,
        seed_offset=100,
        save_best_name="global_best_fedavg.pt",
    )
    metrics_fed.close()

    # ===== 3) Random Cluster =====
    logger.info("=== Running experiment 3/3: Random Cluster ===")
    _save_config(rand_dir, cfg, filename="config.json")

    base_model3, server_state3, clients3 = _init_models_and_clients(cfg, device, client_train, client_test)
    cluster_size = int(cfg.get("rand_cluster_size", 10))
    clusters_by_cid = build_random_clusters(int(cfg["num_users"]), cluster_size, int(cfg["seed"]) + 999)

    # save clusters for record
    with open(os.path.join(rand_dir, "clusters_random.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in clusters_by_cid.items()}, f, ensure_ascii=False, indent=2)

    metrics_rand = MetricsWriter(rand_dir, filename="metrics_randcluster.jsonl")
    run_random_cluster(
        cfg=cfg,
        device=device,
        logger=logger,
        metrics=metrics_rand,
        run_dir=rand_dir,
        base_model=base_model3,
        clients=clients3,
        net_dataidx_map=net_dataidx_map,
        test_dl_global=test_dl_global,
        clusters_by_cid=clusters_by_cid,
        seed_offset=200,
        save_best_name="global_best_randcluster.pt",
    )
    metrics_rand.close()

    # ===== Final compare plots =====
    acc_path = os.path.join(artifacts_dir, "compare_acc.png")
    loss_path = os.path.join(artifacts_dir, "compare_loss.png")

    metrics_items = [
        ("FLIS(HC)", os.path.join(flis_dir, "metrics_flis.jsonl")),
        ("FedAvg", os.path.join(fedavg_dir, "metrics_fedavg.jsonl")),
        ("RandomCluster", os.path.join(rand_dir, "metrics_randcluster.jsonl")),
    ]
    plot_compare_curves(metrics_items, save_acc_path=acc_path, save_loss_path=loss_path)

    logger.info(f"Saved compare accuracy curve: {acc_path}")
    logger.info(f"Saved compare loss curve: {loss_path}")
