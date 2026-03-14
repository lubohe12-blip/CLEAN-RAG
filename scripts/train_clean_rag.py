import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_ext.rag.pipeline import run_clean_rag_pipeline
from src_ext.utils.config import load_config
from src_ext.utils.device import get_device
from src_ext.utils.paths import ensure_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--report_metrics", action="store_true")
    parser.add_argument("--force_clean", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    device = get_device(cfg)

    print("=" * 60)
    print("Start CLEAN+RAG training")
    print("Experiment:", cfg["experiment"]["name"])
    print("Device:", device)
    print("TopK:", cfg["retrieval"]["topk"])
    print("Batch size:", cfg["train"]["batch_size"])
    print("=" * 60)

    outputs = run_clean_rag_pipeline(
        cfg,
        report_metrics=args.report_metrics,
        force_clean=args.force_clean,
    )
    print("CLEAN+RAG baseline finished.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
