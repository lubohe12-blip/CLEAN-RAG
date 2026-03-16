import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the original CLEAN inference pipeline on a local sample CSV."
    )
    parser.add_argument(
        "--test_csv",
        type=Path,
        default=Path("workspace/data/sample/test_sample.csv"),
        help="Path to the tab-delimited sample CSV to predict.",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="split100",
        help="Pretrained CLEAN dataset to use, for example split100 or split70.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sample_test",
        help="Temporary dataset name created under app/data.",
    )
    parser.add_argument(
        "--report_metrics",
        action="store_true",
        help="Print metrics if the input CSV contains real EC labels.",
    )
    parser.add_argument(
        "--keep_app_files",
        action="store_true",
        help="Keep generated app/data CSV and FASTA files after prediction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    app_dir = project_root / "app"
    app_data_dir = app_dir / "data"
    app_pretrained_dir = app_data_dir / "pretrained"
    app_esm_dir = app_dir / "esm"
    workspace_pred_dir = project_root / "workspace" / "outputs" / "predictions"

    test_csv = args.test_csv.resolve()
    if not test_csv.exists():
        raise FileNotFoundError(f"test csv not found: {test_csv}")
    if not app_pretrained_dir.exists():
        raise FileNotFoundError(
            f"missing pretrained directory: {app_pretrained_dir}"
        )
    if not app_esm_dir.exists():
        raise FileNotFoundError(f"missing ESM repo directory: {app_esm_dir}")

    dataset_name = args.dataset_name
    app_csv_path = app_data_dir / f"{dataset_name}.csv"
    app_fasta_path = app_data_dir / f"{dataset_name}.fasta"
    app_result_path = app_dir / "results" / f"{dataset_name}_maxsep.csv"
    workspace_result_path = workspace_pred_dir / f"{dataset_name}_maxsep.csv"

    app_data_dir.mkdir(parents=True, exist_ok=True)
    workspace_pred_dir.mkdir(parents=True, exist_ok=True)
    # Normalize to CLEAN's expected tab-delimited format before inference.
    normalized_df = pd.read_csv(test_csv, sep=None, engine="python")
    normalized_df.to_csv(app_csv_path, index=False, sep="\t")

    original_cwd = Path.cwd()
    try:
        os.chdir(app_dir)
        sys.path.insert(0, str(app_dir / "src"))

        from CLEAN.infer import infer_maxsep
        from CLEAN.utils import csv_to_fasta, retrive_esm1b_embedding

        print(f"Copied test CSV to: {app_csv_path}")
        csv_to_fasta(f"data/{dataset_name}.csv", f"data/{dataset_name}.fasta")
        print(f"Generated FASTA: {app_fasta_path}")

        retrive_esm1b_embedding(dataset_name)
        print(f"Generated ESM embeddings under: {app_dir / 'data' / 'esm_data'}")

        infer_maxsep(
            args.train_data,
            dataset_name,
            report_metrics=args.report_metrics,
            pretrained=True,
        )
    finally:
        os.chdir(original_cwd)

    if not app_result_path.exists():
        raise FileNotFoundError(f"prediction file not produced: {app_result_path}")

    shutil.copyfile(app_result_path, workspace_result_path)
    print(f"Prediction CSV: {app_result_path}")
    print(f"Copied prediction CSV to: {workspace_result_path}")

    if not args.keep_app_files:
        for path in (app_csv_path, app_fasta_path):
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    main()
