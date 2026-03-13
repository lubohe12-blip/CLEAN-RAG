from pathlib import Path


def ensure_dirs(cfg):
    keys = ["output_dir", "log_dir", "ckpt_dir", "pred_dir", "fig_dir", "cache_dir"]
    for key in keys:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)