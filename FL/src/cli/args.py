import argparse
import yaml
from dataclasses import dataclass
from typing import Any, Dict

def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="path to yaml config")
    # allow overriding a few common fields quickly
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--num_users", type=int, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--cluster_alpha", type=float, default=None)
    p.add_argument("--trial", type=int, default=None)
    return p.parse_args()

def load_config(path: str, cli_args) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    override = {}
    for k in ["gpu", "rounds", "num_users", "beta", "cluster_alpha", "trial"]:
        v = getattr(cli_args, k)
        if v is not None:
            override[k] = v
    cfg = _deep_update(cfg, override)
    return cfg
