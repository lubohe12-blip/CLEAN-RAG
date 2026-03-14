import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="原始大csv路径")
    parser.add_argument("--output_dir", type=str, default="workspace/data/sample", help="小样本输出目录")
    parser.add_argument("--train_size", type=int, default=200, help="训练样本数")
    parser.add_argument("--test_size", type=int, default=50, help="测试样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    df = pd.read_csv(input_path)
    print(f"原始数据行数: {len(df)}")
    print(f"列名: {list(df.columns)}")

    if len(df) < args.train_size + args.test_size:
        raise ValueError(
            f"数据总量不足，当前只有 {len(df)} 行，但你想抽 train={args.train_size}, test={args.test_size}"
        )

    # 先整体打乱
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    train_df = df.iloc[:args.train_size].copy()
    test_df = df.iloc[args.train_size:args.train_size + args.test_size].copy()

    train_path = output_dir / "train_sample.csv"
    test_path = output_dir / "test_sample.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"train sample saved to: {train_path}")
    print(f"test sample saved to: {test_path}")
    print(f"train rows: {len(train_df)}")
    print(f"test rows: {len(test_df)}")


if __name__ == "__main__":
    main()