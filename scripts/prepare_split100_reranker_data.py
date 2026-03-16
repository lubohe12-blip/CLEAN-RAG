import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_ext.retrieval.candidate_builder import load_sequence_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="app/data/split100.csv")
    parser.add_argument("--train_out", type=str, default="workspace/data/processed/split100_reranker_train.csv")
    parser.add_argument("--val_out", type=str, default="workspace/data/processed/split100_reranker_val.csv")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input_csv).resolve()
    train_out = (PROJECT_ROOT / args.train_out).resolve()
    val_out = (PROJECT_ROOT / args.val_out).resolve()

    df = load_sequence_table(input_path)
    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    val_size = max(1, int(len(shuffled) * args.val_ratio))
    val_df = shuffled.iloc[:val_size].copy()
    train_df = shuffled.iloc[val_size:].copy()

    if args.max_val_rows > 0:
        val_df = val_df.iloc[: args.max_val_rows].copy()
    if args.max_train_rows > 0:
        train_df = train_df.iloc[: args.max_train_rows].copy()

    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    print("Prepared split100 reranker development data.")
    print(f"input: {input_path}")
    print(f"train_out: {train_out} ({len(train_df)} rows)")
    print(f"val_out: {val_out} ({len(val_df)} rows)")
    if args.max_train_rows > 0 or args.max_val_rows > 0:
        print(
            "sampling_limits: "
            f"max_train_rows={args.max_train_rows}, max_val_rows={args.max_val_rows}"
        )


if __name__ == "__main__":
    main()
