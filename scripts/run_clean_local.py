import argparse
import subprocess
from src_ext.utils.config import load_config
from src_ext.utils.paths import ensure_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    cmd = [
        "python",
        str(cfg["paths"]["app_dir"] / "inference.py")
    ]

    print("Running command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()