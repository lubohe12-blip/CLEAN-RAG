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

- `precision_micro: 0.9978`
- `recall_micro: 0.9388407978923599`
- `f1_micro: 0.9674229203025015`
- `subset_accuracy: 0.9446`
- `num_samples: 5000`
- `num_labels: 1266`

Error analysis summary:

- `same: 4895`
- `clean_wrong_rag_correct: 105`
- `clean_correct_rag_wrong: 0`

## Interpretation

This is the current best no-leakage development baseline.

- It improves over the previous small reranker version by removing the last `clean_correct_rag_wrong` case.
- The gain came from adding candidate-vs-clean-top1 relative features, which reduced false overrides while preserving corrective behavior.
- Keep `new.csv`, `price.csv`, and `halogenase.csv` frozen. Do not use this baseline for held-out final test reporting.
