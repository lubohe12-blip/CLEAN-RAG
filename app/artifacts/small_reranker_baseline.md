# Small Reranker Baseline

This file freezes the current no-leakage development baseline for the split100 small reranker workflow.

## Data Split

- Train file: `workspace/data/processed/split100_reranker_small_train.csv`
- Validation file: `workspace/data/processed/split100_reranker_small_val.csv`
- Scale:
  - train rows: `20000`
  - val rows: `5000`

## Configs

- Train config: `configs/server_reranker_dev_split100_small_baseline_train.yaml`
- Validation config: `configs/server_reranker_dev_split100_small_baseline_val.yaml`

## Model

- Model path:
  `workspace/outputs/checkpoints/clean_server_reranker_dev_split100_small_baseline.pkl`
- Training mode:
  `pairwise`

## Feature Set

Current reranker uses:

- absolute candidate features:
  - `clean_score`
  - `retrieval_score`
  - `base_final_score`
  - `clean_rank_inv`
  - `retrieval_rank_inv`
  - `prototype_rank_inv`
  - `prototype_score`
  - `neighbor_max_score`
  - `neighbor_sum_score`
  - `neighbor_count`
  - `clean_margin`
  - `retrieval_top1`
  - `retrieval_margin`
  - `neighbor_top1`
  - `neighbor_margin`
  - `is_clean_top1`
  - `is_retrieval_top1`
  - `shared_levels_to_clean_top1`
  - `retrieval_advantage_over_clean_top1`
- relative-to-clean-top1 features:
  - `clean_score_gap_to_clean_top1`
  - `retrieval_score_gap_to_clean_top1`
  - `base_final_score_gap_to_clean_top1`
  - `prototype_score_gap_to_clean_top1`
  - `prototype_rank_inv_gap_to_clean_top1`
  - `neighbor_max_gap_to_clean_top1`
  - `neighbor_sum_gap_to_clean_top1`
  - `neighbor_count_gap_to_clean_top1`

## Validation Result

Validation result on `split100_reranker_small_val.csv`:

- `precision_micro: 0.9988`
- `recall_micro: 0.9397817086940158`
- `f1_micro: 0.9683924762458794`
- `subset_accuracy: 0.9456`
- `num_samples: 5000`
- `num_labels: 1262`

Error analysis summary:

- `same: 4890`
- `clean_wrong_rag_correct: 110`
- `clean_correct_rag_wrong: 0`

## Interpretation

This is the current best no-leakage development baseline.

- It improves over the previous pointwise small reranker baseline.
- Compared with the previous pointwise baseline:
  - `f1_micro: 0.9674 -> 0.9684`
  - `subset_accuracy: 0.9446 -> 0.9456`
  - `clean_wrong_rag_correct: 105 -> 110`
  - `clean_correct_rag_wrong: 0 -> 0`
- The gain comes from using a pairwise ranking objective on top of the current relative-feature set.
- Keep `new.csv`, `price.csv`, and `halogenase.csv` frozen. Do not use this baseline for held-out final test reporting.
