import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src_ext.rag.fusion import _build_clean_map, _build_retrieval_map
from src_ext.retrieval.candidate_builder import split_ec_numbers


FEATURE_COLUMNS = [
    "clean_score",
    "retrieval_score",
    "base_final_score",
    "clean_rank_inv",
    "retrieval_rank_inv",
    "prototype_rank_inv",
    "prototype_score",
    "neighbor_max_score",
    "neighbor_sum_score",
    "neighbor_count",
    "clean_margin",
    "retrieval_top1",
    "retrieval_margin",
    "neighbor_top1",
    "neighbor_margin",
    "is_clean_top1",
    "is_retrieval_top1",
    "shared_levels_to_clean_top1",
    "retrieval_advantage_over_clean_top1",
]


def _rank_inverse(rank, default_rank=999):
    rank = int(rank) if rank else default_rank
    return 1.0 / float(rank)


def _shared_ec_levels(left, right):
    left_parts = [part.strip() for part in str(left).split(".") if part.strip()]
    right_parts = [part.strip() for part in str(right).split(".") if part.strip()]

    shared = 0
    for left_part, right_part in zip(left_parts, right_parts):
        if left_part != right_part:
            break
        shared += 1
    return shared


def _prototype_maps(prototype_candidates):
    rank_map = {}
    score_map = {}
    for rank_idx, item in enumerate(prototype_candidates, start=1):
        ec_number = item.get("ec_number")
        if not ec_number:
            continue
        rank_map.setdefault(ec_number, rank_idx)
        score_map[ec_number] = float(item.get("score", 0.0))
    return rank_map, score_map


def _candidate_maps(retrieval_items):
    rank_map = {}
    score_map = _build_retrieval_map(retrieval_items)
    detail_map = {}
    for rank_idx, item in enumerate(retrieval_items, start=1):
        ec_number = item.get("ec_number")
        if not ec_number:
            continue
        rank_map.setdefault(ec_number, rank_idx)
        detail_map[ec_number] = item
    return rank_map, score_map, detail_map


def _clean_maps(clean_items):
    rank_map = {}
    score_map = _build_clean_map(clean_items)
    for rank_idx, item in enumerate(clean_items, start=1):
        ec_number = item.get("ec_number")
        if not ec_number:
            continue
        rank_map.setdefault(ec_number, rank_idx)
    return rank_map, score_map


def _base_ranking_map(fused_payload):
    ranking = fused_payload.get("ranking", [])
    score_map = {}
    rank_map = {}
    detail_map = {}
    for rank_idx, item in enumerate(ranking, start=1):
        ec_number = item.get("ec_number")
        if not ec_number:
            continue
        score_map[ec_number] = float(item.get("final_score", 0.0))
        rank_map.setdefault(ec_number, rank_idx)
        detail_map[ec_number] = item
    return rank_map, score_map, detail_map


def build_candidate_feature_table(
    test_df,
    clean_predictions,
    retrieval_predictions,
    fused_predictions,
    clean_topk=5,
    retrieval_topk=5,
    include_labels=True,
):
    rows = []

    for _, row in test_df.iterrows():
        query_id = row["Entry"]
        true_ecs = set(split_ec_numbers(row["EC number"])) if include_labels else set()
        clean_items = clean_predictions.get(query_id, [])
        retrieval_payload = retrieval_predictions.get(query_id, {})
        retrieval_items = retrieval_payload.get("ec_candidates", [])
        fused_payload = fused_predictions.get(query_id, {})

        clean_rank_map, clean_score_map = _clean_maps(clean_items)
        retrieval_rank_map, retrieval_score_map, retrieval_detail_map = _candidate_maps(retrieval_items)
        prototype_rank_map, prototype_score_map = _prototype_maps(
            retrieval_payload.get("prototype_candidates", [])
        )
        base_rank_map, base_score_map, base_detail_map = _base_ranking_map(fused_payload)

        clean_ranked = [item["ec_number"] for item in clean_items[:clean_topk]]
        retrieval_ranked = [item["ec_number"] for item in retrieval_items[:retrieval_topk]]
        candidate_ecs = []
        seen = set()
        for ec in clean_ranked + retrieval_ranked:
            if ec and ec not in seen:
                seen.add(ec)
                candidate_ecs.append(ec)

        clean_top1_ec = clean_ranked[0] if clean_ranked else ""
        retrieval_top1_ec = retrieval_ranked[0] if retrieval_ranked else ""
        clean_top1_retrieval_score = retrieval_score_map.get(clean_top1_ec, 0.0)

        for ec in candidate_ecs:
            retrieval_detail = retrieval_detail_map.get(ec, {})
            base_detail = base_detail_map.get(ec, {})
            feature_row = {
                "query_id": query_id,
                "ec_number": ec,
                "clean_score": float(clean_score_map.get(ec, 0.0)),
                "retrieval_score": float(retrieval_score_map.get(ec, 0.0)),
                "base_final_score": float(base_score_map.get(ec, 0.0)),
                "clean_rank_inv": _rank_inverse(clean_rank_map.get(ec, 999)),
                "retrieval_rank_inv": _rank_inverse(retrieval_rank_map.get(ec, 999)),
                "prototype_rank_inv": _rank_inverse(prototype_rank_map.get(ec, 999)),
                "prototype_score": float(prototype_score_map.get(ec, 0.0)),
                "neighbor_max_score": float(retrieval_detail.get("neighbor_max_score", 0.0)),
                "neighbor_sum_score": float(retrieval_detail.get("neighbor_sum_score", 0.0)),
                "neighbor_count": float(retrieval_detail.get("neighbor_count", 0.0)),
                "clean_margin": float(fused_payload.get("clean_margin", 0.0)),
                "retrieval_top1": float(fused_payload.get("retrieval_top1", 0.0)),
                "retrieval_margin": float(fused_payload.get("retrieval_margin", 0.0)),
                "neighbor_top1": float(fused_payload.get("neighbor_top1", 0.0)),
                "neighbor_margin": float(fused_payload.get("neighbor_margin", 0.0)),
                "is_clean_top1": float(ec == clean_top1_ec),
                "is_retrieval_top1": float(ec == retrieval_top1_ec),
                "shared_levels_to_clean_top1": float(_shared_ec_levels(ec, clean_top1_ec)),
                "retrieval_advantage_over_clean_top1": float(
                    retrieval_score_map.get(ec, 0.0) - clean_top1_retrieval_score
                ),
                "base_used_retrieval": float(base_detail.get("used_retrieval", False)),
                "label": int(ec in true_ecs) if include_labels else 0,
            }
            rows.append(feature_row)

    columns = ["query_id", "ec_number"] + FEATURE_COLUMNS + ["base_used_retrieval", "label"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


class CandidateReranker:
    VERSION = 1

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.version = self.VERSION
        self.feature_columns = list(FEATURE_COLUMNS)

    def fit(self, feature_df):
        if feature_df.empty:
            raise ValueError("empty feature dataframe")
        y = feature_df["label"].astype(int)
        if y.nunique() < 2:
            raise ValueError("reranker training requires at least two label classes")
        x = feature_df[self.feature_columns]
        self.model.fit(x, y)
        return self

    def predict_scores(self, feature_df):
        if feature_df.empty:
            return pd.Series(dtype=float)
        x = feature_df[self.feature_columns]
        scores = self.model.predict_proba(x)[:, 1]
        return pd.Series(scores, index=feature_df.index, dtype=float)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


def apply_reranker_to_fused_predictions(fused_predictions, feature_df, reranker):
    scored = feature_df.copy()
    scored["reranker_score"] = reranker.predict_scores(scored)

    reranked = {}
    for query_id, group in scored.groupby("query_id", sort=False):
        base_payload = fused_predictions.get(query_id, {})
        base_ranking = {item["ec_number"]: dict(item) for item in base_payload.get("ranking", [])}

        ranking = []
        group = group.sort_values(
            by=["reranker_score", "base_final_score", "retrieval_score", "clean_score"],
            ascending=False,
        )
        for _, row in group.iterrows():
            item = base_ranking.get(
                row["ec_number"],
                {
                    "ec_number": row["ec_number"],
                    "clean_score": float(row.get("clean_score", 0.0)),
                    "retrieval_score": float(row.get("retrieval_score", 0.0)),
                    "used_retrieval": bool(row.get("base_used_retrieval", False)),
                    "clean_margin": float(row.get("clean_margin", 0.0)),
                },
            )
            item = dict(item)
            item.setdefault("clean_score", float(row.get("clean_score", 0.0)))
            item.setdefault("retrieval_score", float(row.get("retrieval_score", 0.0)))
            item.setdefault("used_retrieval", bool(row.get("base_used_retrieval", False)))
            item.setdefault("clean_margin", float(row.get("clean_margin", 0.0)))
            item["final_score"] = float(row["reranker_score"])
            item["base_final_score"] = float(row["base_final_score"])
            item["reranker_score"] = float(row["reranker_score"])
            ranking.append(item)

        reranked[query_id] = dict(base_payload)
        reranked[query_id]["ranking"] = ranking
        reranked[query_id]["reranker_applied"] = True

    return reranked, scored
