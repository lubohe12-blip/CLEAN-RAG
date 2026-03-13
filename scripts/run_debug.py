import argparse
from pathlib import Path
import pandas as pd
import torch

from src_ext.utils.config import load_config
from src_ext.utils.device import get_device
from src_ext.utils.paths import ensure_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    device = get_device(cfg)

    print("=" * 60)
    print("Experiment:", cfg["experiment"]["name"])
    print("Device:", device)
    print("Train file:", cfg["project"]["root"] / cfg["data"]["train_file"])
    print("Test file:", cfg["project"]["root"] / cfg["data"]["test_file"])
    print("=" * 60)

    train_path = cfg["project"]["root"] / cfg["data"]["train_file"]
    test_path = cfg["project"]["root"] / cfg["data"]["test_file"]
    embedding_path = cfg["project"]["root"] / cfg["data"]["embedding_file"]

    if not train_path.exists():
        raise FileNotFoundError(f"train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"train samples: {len(train_df)}")
    print(f"test samples: {len(test_df)}")

    if embedding_path.exists():
        emb = torch.load(embedding_path, map_location="cpu")
        print("embedding loaded.")
        if hasattr(emb, "shape"):
            print("embedding shape:", emb.shape)
    else:
        print("embedding file not found, skip loading:", embedding_path)

    # 这里先做一个假的 retrieval 占位，后面你再接真正逻辑
    topk = cfg["retrieval"]["topk"]
    dummy_results = []
    for i in range(min(len(test_df), 5)):
        dummy_results.append({
            "query_idx": i,
            "topk": topk,
            "pred_ec": "dummy_ec",
            "score": 0.0
        })

    pred_path = cfg["paths"]["pred_dir"] / f'{cfg["experiment"]["name"]}_debug_predictions.csv'
    pd.DataFrame(dummy_results).to_csv(pred_path, index=False)

    print("debug predictions saved to:", pred_path)
    print("run_debug finished successfully.")


if __name__ == "__main__":
    main()