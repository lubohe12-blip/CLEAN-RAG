from pathlib import Path
import yaml


def load_config(config_path: str):
    config_path = Path(config_path).resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg["project"]["root"]).resolve()
    cfg["project"]["root"] = project_root

    for key, rel_path in cfg["paths"].items():
        cfg["paths"][key] = (project_root / rel_path).resolve()

    return cfg