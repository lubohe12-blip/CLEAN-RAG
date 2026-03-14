import subprocess
import sys
from pathlib import Path

import pandas as pd

from src_ext.evaluation.metrics import compute_multilabel_metrics, labels_from_dataframe
from src_ext.evaluation.report import (
    save_metrics_report,
    save_neighbors_report,
    save_prediction_report,
)
from src_ext.rag.fusion import fuse_predictions, parse_clean_prediction_file
from src_ext.retrieval.candidate_builder import build_ec_catalog, build_train_candidates, load_sequence_table
from src_ext.retrieval.retriever import SequenceRetriever
from src_ext.utils.device import get_device


def _run_clean_prediction(cfg, dataset_name, report_metrics=False):
    project_root = Path(cfg["project"]["root"])
    script_path = project_root / "scripts" / "predict_sample_with_clean.py"
    test_csv = project_root / cfg["data"]["test_file"]

    cmd = [
        sys.executable,
        str(script_path),
        "--test_csv",
        str(test_csv),
        "--dataset_name",
        dataset_name,
        "--train_data",
        cfg["retrieval"].get("clean_train_data", "split100"),
    ]
    if report_metrics:
        cmd.append("--report_metrics")

    subprocess.run(cmd, check=True, cwd=project_root)
    return project_root / "workspace" / "outputs" / "predictions" / f"{dataset_name}_maxsep.csv"


def _build_prediction_dataframe(test_df, fused_predictions, retrieval_predictions):
    rows = []
    neighbor_rows = []

    for _, row in test_df.iterrows():
        query_id = row["Entry"]
        fused_payload = fused_predictions.get(query_id, {})
        ranking = fused_payload.get("ranking", [])
        top_prediction = ranking[0]["ec_number"] if ranking else ""
        retrieval_info = retrieval_predictions.get(query_id, {})
        neighbors = retrieval_info.get("neighbors", [])
        prototypes = retrieval_info.get("prototype_candidates", [])

        rows.append(
            {
                "query_id": query_id,
                "true_ec": row["EC number"],
                "pred_ec_top1": top_prediction,
                "pred_ec_ranked": ";".join(item["ec_number"] for item in ranking[:10]),
                "final_scores": ";".join(
                    f"{item['ec_number']}:{item['final_score']:.4f}" for item in ranking[:10]
                ),
                "clean_scores": ";".join(
                    f"{item['ec_number']}:{item['clean_score']:.4f}" for item in ranking[:10]
                ),
                "retrieval_scores": ";".join(
                    f"{item['ec_number']}:{item['retrieval_score']:.4f}" for item in ranking[:10]
                ),
                "support_train_ids": ";".join(neighbor["entry"] for neighbor in neighbors),
                "support_prototype_ecs": ";".join(item["ec_number"] for item in prototypes[:5]),
                "rag_used": bool(fused_payload.get("enable_retrieval", False)),
                "clean_margin": float(fused_payload.get("clean_margin", 0.0)),
                "retrieval_top1": float(fused_payload.get("retrieval_top1", 0.0)),
                "retrieval_margin": float(fused_payload.get("retrieval_margin", 0.0)),
                "neighbor_top1": float(fused_payload.get("neighbor_top1", 0.0)),
                "neighbor_margin": float(fused_payload.get("neighbor_margin", 0.0)),
                "override_gate": bool(fused_payload.get("override_gate", False)),
                "top2_override_applied": bool(fused_payload.get("top2_override_applied", False)),
                "top2_override_reason": fused_payload.get("top2_override_reason", ""),
                "retrieval_top1_override_applied": bool(
                    fused_payload.get("retrieval_top1_override_applied", False)
                ),
                "retrieval_top1_override_reason": fused_payload.get(
                    "retrieval_top1_override_reason", ""
                ),
            }
        )

        neighbor_rows.append(
            {
                "query_id": query_id,
                "neighbors": neighbors,
                "prototype_candidates": prototypes,
                "ec_candidates": retrieval_info.get("ec_candidates", []),
                "enable_retrieval": bool(fused_payload.get("enable_retrieval", False)),
                "clean_margin": float(fused_payload.get("clean_margin", 0.0)),
                "retrieval_top1": float(fused_payload.get("retrieval_top1", 0.0)),
                "retrieval_margin": float(fused_payload.get("retrieval_margin", 0.0)),
                "neighbor_top1": float(fused_payload.get("neighbor_top1", 0.0)),
                "neighbor_margin": float(fused_payload.get("neighbor_margin", 0.0)),
                "override_gate": bool(fused_payload.get("override_gate", False)),
                "top2_override_applied": bool(fused_payload.get("top2_override_applied", False)),
                "top2_override_reason": fused_payload.get("top2_override_reason", ""),
                "retrieval_top1_override_applied": bool(
                    fused_payload.get("retrieval_top1_override_applied", False)
                ),
                "retrieval_top1_override_reason": fused_payload.get(
                    "retrieval_top1_override_reason", ""
                ),
            }
        )

    return pd.DataFrame(rows), neighbor_rows


def run_clean_rag_pipeline(cfg, report_metrics=False, force_clean=False):
    test_df = load_sequence_table(cfg["project"]["root"] / cfg["data"]["test_file"])
    retrieval_train_file = cfg["retrieval"].get("train_file", cfg["data"]["train_file"])
    retrieval_df = load_sequence_table(cfg["project"]["root"] / retrieval_train_file)

    cache_path = cfg["paths"]["cache_dir"] / f"{cfg['experiment']['name']}_retriever.pkl"
    rebuild_retriever = False
    if cache_path.exists():
        retriever = SequenceRetriever.load(cache_path)
        if getattr(retriever, "version", None) != getattr(SequenceRetriever, "VERSION", None):
            rebuild_retriever = True
        if not hasattr(retriever, "prototype_index") or retriever.prototype_index is None:
            rebuild_retriever = True
    else:
        rebuild_retriever = True

    if rebuild_retriever:
        retriever = SequenceRetriever(
            project_root=cfg["project"]["root"],
            clean_train_data=cfg["retrieval"].get("clean_train_data", "split100"),
            device=str(get_device(cfg)),
            prototype_topk=cfg["retrieval"].get("prototype_topk", 5),
            neighbor_max_weight=cfg["retrieval"].get("neighbor_max_weight", 0.40),
            neighbor_sum_weight=cfg["retrieval"].get("neighbor_sum_weight", 0.30),
            neighbor_count_weight=cfg["retrieval"].get("neighbor_count_weight", 0.10),
            prototype_weight=cfg["retrieval"].get("prototype_weight", 0.20),
        )
        if cfg["retrieval"].get("use_clean_precomputed_corpus", False):
            embedding_file = cfg["project"]["root"] / cfg["retrieval"]["embedding_file"]
            retriever.fit_clean_precomputed_corpus(retrieval_df, embedding_file)
        else:
            retriever.fit(
                build_train_candidates(retrieval_df),
                dataset_name=f"{cfg['experiment']['name']}_train_retrieval",
            )
        retriever.save(cache_path)

    retrieval_predictions = retriever.retrieve(
        test_df,
        topk=cfg["retrieval"]["topk"],
        dataset_name=f"{cfg['experiment']['name']}_test_retrieval",
    )

    dataset_name = cfg["experiment"].get("clean_dataset_name", "sample_test_rag")
    clean_pred_path = cfg["paths"]["pred_dir"] / f"{dataset_name}_maxsep.csv"
    if force_clean or not clean_pred_path.exists():
        clean_pred_path = _run_clean_prediction(cfg, dataset_name, report_metrics=report_metrics)

    clean_predictions = parse_clean_prediction_file(clean_pred_path)
    fused_predictions = fuse_predictions(
        clean_predictions,
        retrieval_predictions,
        clean_weight=cfg["retrieval"].get("clean_weight", 0.85),
        retrieval_weight=cfg["retrieval"].get("retrieval_weight", 0.15),
        rerank_topk=cfg["retrieval"].get("rerank_topk", 5),
        margin_threshold=cfg["retrieval"].get("margin_threshold", 0.08),
        min_retrieval_score=cfg["retrieval"].get("min_retrieval_score", 0.45),
        min_retrieval_margin=cfg["retrieval"].get("min_retrieval_margin", 0.05),
        override_retrieval_score=cfg["retrieval"].get("override_retrieval_score", 0.80),
        override_retrieval_margin=cfg["retrieval"].get("override_retrieval_margin", 0.15),
        top2_override_enabled=cfg["retrieval"].get("top2_override_enabled", True),
        top2_clean_gap_max=cfg["retrieval"].get("top2_clean_gap_max", 0.08),
        top2_retrieval_advantage_min=cfg["retrieval"].get("top2_retrieval_advantage_min", 0.10),
        top2_require_retrieval_top2_match=cfg["retrieval"].get("top2_require_retrieval_top2_match", True),
        retrieval_top1_override_enabled=cfg["retrieval"].get("retrieval_top1_override_enabled", True),
        retrieval_top1_max_clean_candidates=cfg["retrieval"].get("retrieval_top1_max_clean_candidates", 1),
        retrieval_top1_min_score=cfg["retrieval"].get("retrieval_top1_min_score", 0.95),
        retrieval_top1_min_margin=cfg["retrieval"].get("retrieval_top1_min_margin", 0.50),
        allow_new_ecs=cfg["retrieval"].get("allow_new_ecs", True),
        max_new_ecs=cfg["retrieval"].get("max_new_ecs", 2),
    )

    prediction_df, neighbor_rows = _build_prediction_dataframe(
        test_df, fused_predictions, retrieval_predictions
    )
    ec_catalog = build_ec_catalog(retrieval_df)

    pred_path = cfg["paths"]["pred_dir"] / f"{cfg['experiment']['name']}_rag_predictions.csv"
    neighbors_path = cfg["paths"]["pred_dir"] / f"{cfg['experiment']['name']}_rag_neighbors.json"
    metrics_path = cfg["paths"]["pred_dir"] / f"{cfg['experiment']['name']}_rag_metrics.json"
    catalog_path = cfg["paths"]["pred_dir"] / f"{cfg['experiment']['name']}_ec_catalog.csv"

    save_prediction_report(prediction_df, pred_path)
    save_neighbors_report(neighbor_rows, neighbors_path)
    save_prediction_report(ec_catalog, catalog_path)

    metrics = None
    if report_metrics:
        true_labels = labels_from_dataframe(test_df)
        pred_labels = [
            row["pred_ec_ranked"].split(";")[:1] if row["pred_ec_ranked"] else []
            for _, row in prediction_df.iterrows()
        ]
        metrics = compute_multilabel_metrics(true_labels, pred_labels)
        save_metrics_report(metrics, metrics_path)

    return {
        "predictions": pred_path,
        "neighbors": neighbors_path,
        "catalog": catalog_path,
        "metrics": metrics,
        "clean_predictions": clean_pred_path,
        "retriever_cache": cache_path,
    }
