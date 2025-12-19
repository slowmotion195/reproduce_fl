from src.cli.args import parse_args, load_config
from src.core.seed import set_seed
from src.core.device import build_device
from src.core.logging import make_run_dir, build_logger
from src.core.metrics import MetricsWriter

from src.fl.algorithms.flis_hc import run_flis_hc
from src.fl.algorithms.compare import run_compare


def main():
    args = parse_args()
    cfg = load_config(args.config, args)

    set_seed(int(cfg["seed"]))
    device = build_device(int(cfg["gpu"]))

    run_dir = make_run_dir(
        out_root=cfg["out_root"],
        alg=cfg["alg"],
        dataset=cfg["dataset"],
        trial=int(cfg["trial"])
    )
    logger = build_logger(run_dir)

    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Device: {device}")

    alg = cfg.get("alg", "flis_hc")

    if alg == "flis_hc":
        metrics = MetricsWriter(run_dir)
        run_flis_hc(cfg=cfg, device=device, logger=logger, metrics=metrics, run_dir=run_dir)
        metrics.close()

    elif alg in ["compare", "compare3"]:
        # compare 会内部创建三份 metrics 文件并绘图
        run_compare(cfg=cfg, device=device, logger=logger, run_dir=run_dir)

    else:
        raise ValueError(f"Unsupported alg: {alg}")
