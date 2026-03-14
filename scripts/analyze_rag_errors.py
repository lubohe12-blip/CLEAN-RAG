import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_ext.retrieval.candidate_builder import load_sequence_table, split_ec_numbers
from src_ext.rag.fusion import parse_clean_prediction_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def _top1_from_clean_predictions(clean_pred_map):
    return {
        query_id: (items[0]["ec_number"] if items else "")
        for query_id, items in clean_pred_map.items()
    }


def _is_correct(true_ecs, pred_ec):
    return pred_ec in set(true_ecs)


def _optional_column_as_dict(df, key_col, value_col, default):
    if value_col not in df.columns:
        return {key: default for key in df[key_col].tolist()}
    return dict(zip(df[key_col], df[value_col]))


def main():
    args = parse_args()

    from src_ext.utils.config import load_config

    cfg = load_config(args.config)
    project_root = Path(cfg["project"]["root"])
    test_df = load_sequence_table(project_root / cfg["data"]["test_file"])

    clean_pred_path = cfg["paths"]["pred_dir"] / f"{cfg['experiment']['clean_dataset_name']}_maxsep.csv"
    rag_pred_path = cfg["paths"]["pred_dir"] / f"{cfg['experiment']['name']}_rag_predictions.csv"
    out_path = cfg["paths"]["pred_dir"] / f"{cfg['experiment']['name']}_rag_error_analysis.csv"

    clean_top1 = _top1_from_clean_predictions(parse_clean_prediction_file(clean_pred_path))
    rag_df = pd.read_csv(rag_pred_path)
    rag_top1 = dict(zip(rag_df["query_id"], rag_df["pred_ec_top1"]))
    rag_used = _optional_column_as_dict(rag_df, "query_id", "rag_used", False)
    clean_margin = _optional_column_as_dict(rag_df, "query_id", "clean_margin", 0.0)
    retrieval_top1 = _optional_column_as_dict(rag_df, "query_id", "retrieval_top1", 0.0)
    retrieval_margin = _optional_column_as_dict(rag_df, "query_id", "retrieval_margin", 0.0)
    neighbor_top1 = _optional_column_as_dict(rag_df, "query_id", "neighbor_top1", 0.0)
    neighbor_margin = _optional_column_as_dict(rag_df, "query_id", "neighbor_margin", 0.0)
    override_gate = _optional_column_as_dict(rag_df, "query_id", "override_gate", False)
    top2_override_applied = _optional_column_as_dict(rag_df, "query_id", "top2_override_applied", False)
    top2_override_reason = _optional_column_as_dict(rag_df, "query_id", "top2_override_reason", "")
    retrieval_top1_override_applied = _optional_column_as_dict(
        rag_df, "query_id", "retrieval_top1_override_applied", False
    )
    retrieval_top1_override_reason = _optional_column_as_dict(
        rag_df, "query_id", "retrieval_top1_override_reason", ""
    )
    support_ids = _optional_column_as_dict(rag_df, "query_id", "support_train_ids", "")
    support_prototypes = _optional_column_as_dict(rag_df, "query_id", "support_prototype_ecs", "")

    records = []
    for _, row in test_df.iterrows():
        query_id = row["Entry"]
        true_ecs = split_ec_numbers(row["EC number"])
        clean_pred = clean_top1.get(query_id, "")
        rag_pred = rag_top1.get(query_id, "")

        clean_ok = _is_correct(true_ecs, clean_pred)
        rag_ok = _is_correct(true_ecs, rag_pred)

        if clean_ok == rag_ok:
            delta = "same"
        elif clean_ok and not rag_ok:
            delta = "clean_correct_rag_wrong"
        else:
            delta = "clean_wrong_rag_correct"

        records.append(
            {
                "query_id": query_id,
                "true_ec": ";".join(true_ecs),
                "clean_top1": clean_pred,
                "rag_top1": rag_pred,
                "clean_correct": clean_ok,
                "rag_correct": rag_ok,
                "delta": delta,
                "rag_used": bool(rag_used.get(query_id, False)),
                "clean_margin": float(clean_margin.get(query_id, 0.0)),
                "retrieval_top1": float(retrieval_top1.get(query_id, 0.0)),
                "retrieval_margin": float(retrieval_margin.get(query_id, 0.0)),
                "neighbor_top1": float(neighbor_top1.get(query_id, 0.0)),
                "neighbor_margin": float(neighbor_margin.get(query_id, 0.0)),
                "override_gate": bool(override_gate.get(query_id, False)),
                "top2_override_applied": bool(top2_override_applied.get(query_id, False)),
                "top2_override_reason": top2_override_reason.get(query_id, ""),
                "retrieval_top1_override_applied": bool(
                    retrieval_top1_override_applied.get(query_id, False)
                ),
                "retrieval_top1_override_reason": retrieval_top1_override_reason.get(query_id, ""),
                "support_train_ids": support_ids.get(query_id, ""),
                "support_prototype_ecs": support_prototypes.get(query_id, ""),
            }
        )

    out_df = pd.DataFrame(records)
    out_df.to_csv(out_path, index=False)

    summary = out_df["delta"].value_counts().to_dict()
    print("error analysis saved to:", out_path)
    print("summary:", summary)
    if "rag_used" not in rag_df.columns:
        print("note: rag_predictions.csv is from the old format; rerun evaluate_experiment.py to get gated-fusion diagnostics.")


if __name__ == "__main__":
    main()
