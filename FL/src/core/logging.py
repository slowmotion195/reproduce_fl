import os
import time
import logging
from pathlib import Path

def make_run_dir(out_root: str, alg: str, dataset: str, trial: int) -> str:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{ts}_{alg}_{dataset}_trial{trial}"
    path = Path(out_root) / run_name
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    (path / "artifacts").mkdir(exist_ok=True)
    return str(path)

def build_logger(run_dir: str) -> logging.Logger:
    logger = logging.getLogger("flis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(run_dir, "logs.txt"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
