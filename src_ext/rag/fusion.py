import csv
from pathlib import Path


def parse_clean_prediction_file(path):
    path = Path(path)
    if not path.exists():
        return {}

    predictions = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            query_id = row[0].strip()
            ec_items = []
            for rank_idx, raw in enumerate(row[1:]):
                raw = raw.strip()
                if not raw.startswith("EC:"):
                    continue
                try:
                    ec_part, dist_part = raw.split("/", 1)
                    ec_number = ec_part.replace("EC:", "").strip()
                    distance = float(dist_part)
                    ec_items.append(
                        {
                            "ec_number": ec_number,
                            "distance": distance,
                            "rank": rank_idx,
                        }
                    )
                except ValueError:
                    continue
            predictions[query_id] = ec_items
    return predictions


def _distance_to_score(distance):
    return 1.0 / (1.0 + max(float(distance), 0.0))


def _normalize(values):
    if not values:
        return {}
    max_value = max(values.values())
    if max_value <= 0.0:
        max_value = 1.0
    return {key: value / max_value for key, value in values.items()}


def _clean_margin(clean_items):
    if len(clean_items) < 2:
        return 1.0

    top1 = _distance_to_score(clean_items[0]["distance"])
    top2 = _distance_to_score(clean_items[1]["distance"])
    return max(top1 - top2, 0.0)


def _build_retrieval_map(retrieval_items):
    raw = {item["ec_number"]: max(float(item["score"]), 0.0) for item in retrieval_items}
    return _normalize(raw)


def _build_clean_map(clean_items):
    raw = {item["ec_number"]: _distance_to_score(item["distance"]) for item in clean_items}
    return _normalize(raw)


def _shared_ec_levels(left, right):
    left_parts = [part.strip() for part in str(left).split(".") if part.strip()]
    right_parts = [part.strip() for part in str(right).split(".") if part.strip()]

    shared = 0
    for left_part, right_part in zip(left_parts, right_parts):
        if left_part != right_part:
            break
        shared += 1
    return shared


def _prototype_rank_map(prototype_candidates):
    ranks = {}
    for rank_idx, candidate in enumerate(prototype_candidates, start=1):
        ec_number = candidate.get("ec_number")
        if ec_number and ec_number not in ranks:
            ranks[ec_number] = rank_idx
    return ranks


def fuse_predictions(
    clean_predictions,
    retrieval_predictions,
    clean_weight=0.85,
    retrieval_weight=0.15,
    rerank_topk=5,
    rerank_max_prototype_rank=0,
    rerank_require_prototype_not_worse_than_clean=False,
    margin_threshold=0.08,
    min_retrieval_score=0.45,
    min_retrieval_margin=0.05,
    override_retrieval_score=0.80,
    override_retrieval_margin=0.15,
    top2_override_enabled=True,
    top2_clean_gap_max=0.08,
    top2_retrieval_advantage_min=0.10,
    top2_require_retrieval_top2_match=True,
    retrieval_top1_override_enabled=True,
    retrieval_top1_max_clean_candidates=1,
    retrieval_top1_min_score=0.95,
    retrieval_top1_min_margin=0.50,
    retrieval_top1_min_shared_levels=0,
    retrieval_top1_min_clean_advantage=0.0,
    retrieval_top1_max_prototype_rank=0,
    allow_new_ecs=True,
    max_new_ecs=2,
):
    fused = {}
    query_ids = set(clean_predictions) | set(retrieval_predictions)

    for query_id in query_ids:
        clean_items = clean_predictions.get(query_id, [])
        retrieval_payload = retrieval_predictions.get(query_id, {})
        retrieval_items = retrieval_payload.get("ec_candidates", [])
        prototype_rank_map = _prototype_rank_map(retrieval_payload.get("prototype_candidates", []))

        clean_scores = _build_clean_map(clean_items)
        retrieval_scores = _build_retrieval_map(retrieval_items)
        margin = _clean_margin(clean_items)
        retrieval_top1 = float(retrieval_payload.get("ec_top1_score", 0.0))
        retrieval_margin = float(retrieval_payload.get("ec_margin_score", 0.0))
        neighbor_top1 = float(retrieval_payload.get("retrieval_top1_raw", 0.0))
        neighbor_margin = float(retrieval_payload.get("retrieval_margin_raw", 0.0))

        standard_gate = (
            margin < margin_threshold
            and retrieval_top1 >= min_retrieval_score
            and retrieval_margin >= min_retrieval_margin
        )
        override_gate = (
            retrieval_top1 >= override_retrieval_score
            and retrieval_margin >= override_retrieval_margin
        )
        enable_retrieval = standard_gate or override_gate

        clean_rank_order = [item["ec_number"] for item in clean_items]
        clean_top1_ec = clean_rank_order[0] if clean_rank_order else ""
        clean_top1_prototype_rank = prototype_rank_map.get(clean_top1_ec, 0)
        rerank_pool = set(clean_rank_order[:rerank_topk]) if rerank_topk > 0 else set(clean_rank_order)
        if allow_new_ecs and enable_retrieval:
            retrieval_ranked_ecs = [item["ec_number"] for item in retrieval_items[:max_new_ecs]]
            rerank_pool |= set(retrieval_ranked_ecs)

        ranking = []
        seen = set()
        for ec in clean_rank_order:
            prototype_rank = prototype_rank_map.get(ec, 0)
            prototype_ok = True
            if ec != clean_top1_ec:
                if rerank_max_prototype_rank > 0:
                    prototype_ok = prototype_rank > 0 and prototype_rank <= rerank_max_prototype_rank
                if (
                    prototype_ok
                    and rerank_require_prototype_not_worse_than_clean
                    and clean_top1_prototype_rank > 0
                ):
                    prototype_ok = prototype_rank > 0 and prototype_rank <= clean_top1_prototype_rank

            if ec in rerank_pool and enable_retrieval and prototype_ok:
                final_score = clean_weight * clean_scores.get(ec, 0.0) + retrieval_weight * retrieval_scores.get(
                    ec, 0.0
                )
            else:
                final_score = clean_scores.get(ec, 0.0)

            ranking.append(
                {
                    "ec_number": ec,
                    "final_score": final_score,
                    "clean_score": clean_scores.get(ec, 0.0),
                    "retrieval_score": retrieval_scores.get(ec, 0.0),
                    "used_retrieval": bool(enable_retrieval and ec in rerank_pool and prototype_ok),
                    "clean_margin": margin,
                }
            )
            seen.add(ec)

        if allow_new_ecs and enable_retrieval:
            for ec, retrieval_score in retrieval_scores.items():
                if ec in seen:
                    continue
                prototype_rank = prototype_rank_map.get(ec, 0)
                prototype_ok = True
                if rerank_max_prototype_rank > 0:
                    prototype_ok = prototype_rank > 0 and prototype_rank <= rerank_max_prototype_rank
                if (
                    prototype_ok
                    and rerank_require_prototype_not_worse_than_clean
                    and clean_top1_prototype_rank > 0
                ):
                    prototype_ok = prototype_rank > 0 and prototype_rank <= clean_top1_prototype_rank
                ranking.append(
                    {
                        "ec_number": ec,
                        "final_score": (
                            clean_weight * clean_scores.get(ec, 0.0)
                            + retrieval_weight * retrieval_score
                            if prototype_ok
                            else clean_scores.get(ec, 0.0)
                        ),
                        "clean_score": clean_scores.get(ec, 0.0),
                        "retrieval_score": retrieval_score,
                        "used_retrieval": bool(prototype_ok),
                        "clean_margin": margin,
                    }
                )

        ranking.sort(key=lambda item: item["final_score"], reverse=True)
        override_applied = False
        override_reason = ""
        retrieval_top1_override_applied = False
        retrieval_top1_override_reason = ""

        if top2_override_enabled and len(ranking) >= 2 and len(clean_rank_order) >= 2:
            clean_top2_ec = clean_rank_order[1]
            ranking_map = {item["ec_number"]: item for item in ranking}
            top1_item = ranking_map.get(clean_top1_ec)
            top2_item = ranking_map.get(clean_top2_ec)

            retrieval_ranked_ecs = [item["ec_number"] for item in retrieval_items]
            retrieval_top1_ec = retrieval_ranked_ecs[0] if retrieval_ranked_ecs else ""
            retrieval_top2_ec = retrieval_ranked_ecs[1] if len(retrieval_ranked_ecs) > 1 else ""

            clean_gap = 0.0
            retrieval_advantage = 0.0
            if top1_item and top2_item:
                clean_gap = top1_item["clean_score"] - top2_item["clean_score"]
                retrieval_advantage = top2_item["retrieval_score"] - top1_item["retrieval_score"]

            retrieval_alignment_ok = retrieval_top1_ec == clean_top2_ec
            if top2_require_retrieval_top2_match:
                retrieval_alignment_ok = retrieval_alignment_ok or retrieval_top2_ec == clean_top2_ec

            if (
                override_gate
                and retrieval_alignment_ok
                and top1_item is not None
                and top2_item is not None
                and clean_gap <= top2_clean_gap_max
                and retrieval_advantage >= top2_retrieval_advantage_min
            ):
                top2_item["final_score"] = max(top2_item["final_score"], top1_item["final_score"] + 1e-6)
                override_applied = True
                override_reason = (
                    f"promote_clean_top2:{clean_top2_ec}|clean_gap={clean_gap:.4f}|"
                    f"retrieval_advantage={retrieval_advantage:.4f}"
                )
                ranking.sort(key=lambda item: item["final_score"], reverse=True)

        if retrieval_top1_override_enabled and not override_applied:
            retrieval_top1_ec = retrieval_items[0]["ec_number"] if retrieval_items else ""
            shared_levels = _shared_ec_levels(clean_top1_ec, retrieval_top1_ec)
            clean_top1_retrieval_score = retrieval_scores.get(clean_top1_ec, 0.0)
            retrieval_clean_advantage = retrieval_scores.get(retrieval_top1_ec, 0.0) - clean_top1_retrieval_score
            prototype_rank = 0
            for rank_idx, candidate in enumerate(retrieval_payload.get("prototype_candidates", []), start=1):
                if candidate.get("ec_number") == retrieval_top1_ec:
                    prototype_rank = rank_idx
                    break
            if (
                retrieval_top1_ec
                and clean_top1_ec
                and retrieval_top1_ec != clean_top1_ec
                and len(clean_rank_order) <= retrieval_top1_max_clean_candidates
                and retrieval_top1 >= retrieval_top1_min_score
                and retrieval_margin >= retrieval_top1_min_margin
                and shared_levels >= retrieval_top1_min_shared_levels
                and retrieval_clean_advantage >= retrieval_top1_min_clean_advantage
                and (
                    retrieval_top1_max_prototype_rank <= 0
                    or (prototype_rank > 0 and prototype_rank <= retrieval_top1_max_prototype_rank)
                )
            ):
                ranking_map = {item["ec_number"]: item for item in ranking}
                top1_item = ranking_map.get(clean_top1_ec)
                retrieval_item = ranking_map.get(retrieval_top1_ec)
                if retrieval_item is None:
                    retrieval_item = {
                        "ec_number": retrieval_top1_ec,
                        "final_score": retrieval_weight * retrieval_scores.get(retrieval_top1_ec, 0.0),
                        "clean_score": clean_scores.get(retrieval_top1_ec, 0.0),
                        "retrieval_score": retrieval_scores.get(retrieval_top1_ec, 0.0),
                        "used_retrieval": True,
                        "clean_margin": margin,
                    }
                    ranking.append(retrieval_item)
                if top1_item is not None:
                    retrieval_item["final_score"] = max(retrieval_item["final_score"], top1_item["final_score"] + 1e-6)
                    retrieval_top1_override_applied = True
                    retrieval_top1_override_reason = (
                        f"promote_retrieval_top1:{retrieval_top1_ec}|clean_candidates={len(clean_rank_order)}|"
                        f"retrieval_score={retrieval_top1:.4f}|retrieval_margin={retrieval_margin:.4f}|"
                        f"shared_levels={shared_levels}|clean_advantage={retrieval_clean_advantage:.4f}|"
                        f"prototype_rank={prototype_rank}"
                    )
                    ranking.sort(key=lambda item: item["final_score"], reverse=True)

        fused[query_id] = {
            "ranking": ranking,
            "enable_retrieval": enable_retrieval,
            "clean_margin": margin,
            "retrieval_top1": retrieval_top1,
            "retrieval_margin": retrieval_margin,
            "neighbor_top1": neighbor_top1,
            "neighbor_margin": neighbor_margin,
            "override_gate": override_gate,
            "top2_override_applied": override_applied,
            "top2_override_reason": override_reason,
            "retrieval_top1_override_applied": retrieval_top1_override_applied,
            "retrieval_top1_override_reason": retrieval_top1_override_reason,
        }
    return fused
