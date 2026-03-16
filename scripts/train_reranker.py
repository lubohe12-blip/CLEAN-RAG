import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_ext.evaluation.metrics import compute_multilabel_metrics, labels_from_dataframe
from src_ext.rag.pipeline import _resolve_reranker_path, run_clean_rag_pipeline
from src_ext.rag.reranker import CandidateReranker
from src_ext.retrieval.candidate_builder import load_sequence_table, split_ec_numbers
from src_ext.utils.config import load_config
from src_ext.utils.paths import ensure_dirs


DEFAULT_HELD_OUT_FILES = {"new.csv", "price.csv", "halogenase.csv"}


def _top1_labels(feature_df):
    pred_labels = []
    for _, group in feature_df.groupby("query_id", sort=False):
        group = group.sort_values(
            by=["reranker_score", "base_final_score", "retrieval_score", "clean_score"],
            ascending=False,
        )
        top_ec = group.iloc[0]["ec_number"] if not group.empty else ""
        pred_labels.append([top_ec] if top_ec else [])
    return pred_labels


def _held_out_filenames(cfg):
    policy = cfg.get("policy", {})
    return {
        Path(item).name
        for item in policy.get("held_out_test_files", sorted(DEFAULT_HELD_OUT_FILES))
    }


def _assert_reranker_training_allowed(cfg):
    held_out = _held_out_filenames(cfg)
    test_filename = Path(cfg["data"]["test_file"]).name
    if test_filename in held_out:
        raise ValueError(
            "reranker training is blocked for held-out final test sets: "
            f"{sorted(held_out)}. Current test_file is {test_filename}. "
            "Use a separate training/validation config instead."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--force_clean", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    _assert_reranker_training_allowed(cfg)

    cfg.setdefault("reranker", {})
    cfg["reranker"]["enabled"] = False

    outputs = run_clean_rag_pipeline(
        cfg,
        report_metrics=False,
        force_clean=args.force_clean,
    )

    feature_df = pd.read_csv(outputs["reranker_features"])
    reranker_cfg = cfg.get("reranker", {})
    reranker = CandidateReranker(
        training_mode=reranker_cfg.get("training_mode", "pairwise"),
        max_negatives_per_positive=reranker_cfg.get("max_negatives_per_positive", 5),
    ).fit(feature_df)
    model_path = _resolve_reranker_path(cfg)
    reranker.save(model_path)

    scored_df = feature_df.copy()
    scored_df["reranker_score"] = reranker.predict_scores(scored_df)

    test_df = load_sequence_table(cfg["project"]["root"] / cfg["data"]["test_file"])
    true_labels = labels_from_dataframe(test_df)
    pred_labels = _top1_labels(scored_df)
    metrics = compute_multilabel_metrics(true_labels, pred_labels)

    train_metrics_path = cfg["paths"]["pred_dir"] / f"{cfg['experiment']['name']}_reranker_train_metrics.json"
    with open(train_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    scored_path = cfg["paths"]["pred_dir"] / f"{cfg['experiment']['name']}_reranker_train_features.csv"
    scored_df.to_csv(scored_path, index=False)

    print("Reranker training finished.")
    print(f"model: {model_path}")
    print(f"features: {scored_path}")
    print(f"metrics: {metrics}")


if __name__ == "__main__":
    main()
