# Reranker Development Log And Roadmap

This document tracks the development-stage reranker work under the no-leakage policy:

- held-out final test sets remain frozen:
  - `app/data/new.csv`
  - `app/data/price.csv`
  - `app/data/datasets/halogenase.csv`
- reranker development currently uses split100-derived train/validation subsets only

## 1. Objective

Move from rule-based RAG reranking to a learned reranker that can:

- preserve the correction gains from retrieval/prototype evidence
- reduce hand-tuned override logic
- generalize better than the current rule stack on held-out final test sets

## 2. Frozen Rule Baselines

### 2.1 Sample-set rule baseline

On `workspace/data/sample/test_sample.csv`:

- original CLEAN errors: `4`
- final rule-based CLEAN+RAG errors: `1`
- summary:
  - `clean_wrong_rag_correct: 3`
  - `clean_correct_rag_wrong: 0`

This established that retrieval + prototype + controlled reranking is useful.

### 2.2 Full-eval rule baseline on `new.csv`

This result is kept as the current held-out rule baseline and must not be used for reranker development:

- `precision_micro: 0.5714`
- `recall_micro: 0.4453`
- `f1_micro: 0.5006`
- `subset_accuracy: 0.4719`
- `clean_wrong_rag_correct: 14`
- `clean_correct_rag_wrong: 2`

Main conclusions from full-eval ablation:

- `top2_override` did not help on the full test set
- useful gain came mainly from `retrieval_top1_override`
- stricter gating improved stability

## 3. Reranker Development Policy

### 3.1 What is forbidden during development

The following files are frozen for final testing only and must not be used to train or validate the reranker:

- `app/data/new.csv`
- `app/data/price.csv`
- `app/data/datasets/halogenase.csv`

### 3.2 Current development data

Current development is performed on split100-derived subsets:

- source corpus: `app/data/split100.csv`
- retrieval corpus: `split100`
- precomputed embeddings: `app/data/pretrained/100.pt`

Two development scales were defined:

- small:
  - train: `20000`
  - val: `5000`
- mid:
  - train: `50000`
  - val: `10000`
  - currently too memory-heavy on the available server and not used as the active baseline

## 4. Reranker Implementation Status

### 4.1 Implemented components

- candidate feature extraction:
  - `src_ext/rag/reranker.py`
- reranker training entry:
  - `scripts/train_reranker.py`
- reranker-aware pipeline:
  - `src_ext/rag/pipeline.py`
- split100 dev split builder:
  - `scripts/prepare_split100_reranker_data.py`

### 4.2 Current model form

Current reranker is a pairwise `LogisticRegression` reranker over candidate EC rows.

Training now uses query-local positive-vs-negative candidate differences.

This replaced the earlier pointwise candidate classifier because the task is a ranking problem rather than an independent per-candidate classification problem.

### 4.3 Feature evolution

#### Initial feature set

The first reranker used absolute candidate features:

- clean score
- retrieval score
- base fused score
- clean/retrieval/prototype ranks
- prototype score
- neighbor max / sum / count
- clean margin
- retrieval top1 / margin
- neighbor top1 / margin
- identity flags (`is_clean_top1`, `is_retrieval_top1`)
- EC prefix overlap with clean top1
- retrieval advantage over clean top1

This version worked, but one persistent false override remained on small validation.

#### Current feature set

The current version added candidate-vs-clean-top1 relative features:

- `clean_score_gap_to_clean_top1`
- `retrieval_score_gap_to_clean_top1`
- `base_final_score_gap_to_clean_top1`
- `prototype_score_gap_to_clean_top1`
- `prototype_rank_inv_gap_to_clean_top1`
- `neighbor_max_gap_to_clean_top1`
- `neighbor_sum_gap_to_clean_top1`
- `neighbor_count_gap_to_clean_top1`

This change removed the remaining false positive on small validation for the pointwise reranker and became the feature base for the pairwise reranker.

#### Current objective

The current small baseline no longer uses pointwise training.

It uses a minimal pairwise objective:

- within each `query_id`, build positive-vs-negative feature differences
- train `LogisticRegression` on those pairwise difference rows
- use the learned decision function as the reranking score

## 5. Small Dev Baseline

### 5.1 Configs

- train:
  - `configs/server_reranker_dev_split100_small_baseline_train.yaml`
- validation:
  - `configs/server_reranker_dev_split100_small_baseline_val.yaml`

### 5.2 Model artifact

- `workspace/outputs/checkpoints/clean_server_reranker_dev_split100_small_baseline.pkl`
- training mode: `pairwise`

### 5.3 Validation result

Validation on `split100_reranker_small_val.csv`:

- `precision_micro: 0.9988`
- `recall_micro: 0.9397817086940158`
- `f1_micro: 0.9683924762458794`
- `subset_accuracy: 0.9456`
- `num_samples: 5000`
- `num_labels: 1262`

Error analysis:

- `same: 4890`
- `clean_wrong_rag_correct: 110`
- `clean_correct_rag_wrong: 0`

### 5.4 Interpretation

This is the current best no-leakage reranker development baseline.

Important meaning:

- the learned reranker now produces net correction gain on validation
- the previous lone bad override case remains removed
- moving from pointwise to pairwise training adds another `5` corrected cases without introducing new errors
- the relative features plus pairwise objective now define the active small development baseline

## 6. Representative Error Pattern

### 6.1 Previously observed failure

The most important small-val failure before the relative-feature update was:

- `P76149`
  - true EC: `1.2.1.16`
  - CLEAN top1: `1.2.1.16`
  - RAG/reranker top1: `1.2.1.79`

This was an override-style error where:

- retrieval and neighbor evidence preferred a wrong EC
- prototype still ranked the clean truth strongly
- the earlier reranker did not model candidate-vs-clean-top1 relative structure well enough

This failure is no longer present in the current small baseline.

### 6.2 Representative corrected cases

Examples of corrected cases from small validation include:

- `Q4DPN8`: `2.5.1.114 -> 2.1.1.228`
- `O25613`: `2.5.1.3 -> 3.5.4.16`
- `Q9SJ61`: `2.7.11.17 -> 2.7.11.1`
- `Q93V61`: `2.3.1.43 -> 3.1.1.32`
- `Q9ZWB3`: `2.7.11.26 -> 2.7.11.1`
- `Q9HNQ0`: `3.5.1.90 -> 4.1.1.19`
- `P97855`: `2.3.1.48 -> 3.6.4.12`
- `Q89NA7`: `4.6.1.1 -> 3.5.1.2`
- `B8I4G1`: `4.2.1.85 -> 4.2.1.33`
- `P18786`: `3.1.26.5 -> 1.1.1.23`

This suggests the reranker is learning more than one narrow rule:

- it can fix fine-grained same-family EC confusions
- it can also fix some broader CLEAN misclassifications

## 7. Open Risks

- the current reranker is pairwise, but still not listwise or fully query-grouped in the optimization sense
- validation scale is still only `20k/5k`
- the `50k/10k` mid-scale development split currently exceeds the available server memory budget
- held-out final evaluation has not yet been rerun with a finalized reranker, which is correct at this stage

## 8. Next-Step Roadmap

### Phase A: stabilize small-scale reranker development

Immediate next work should stay on the `20k/5k` split:

1. freeze the current small baseline
2. keep all future changes compared against this baseline
3. avoid touching `new.csv`, `price.csv`, and `halogenase.csv`

### Phase B: next reranker iteration

Most promising next direction:

1. keep pairwise training as the current default
2. next consider a stronger query-aware reranker:
   - better negative sampling
   - grouped ranking losses
   - listwise objectives
3. preserve the current feature set as the starting feature backbone

### Phase C: scale cautiously

After the next reranker variant beats the current small baseline:

1. retry a medium-scale split
2. reduce memory pressure by controlling:
   - train rows
   - batch size
   - worker count
3. do not jump directly to full-scale split100 development

### Phase D: final testing

Only after the reranker architecture and hyperparameters stop changing:

1. unfreeze held-out evaluation
2. test on:
   - `new.csv`
   - `price.csv`
   - `halogenase.csv`
3. compare against:
   - original CLEAN
   - best rule-based RAG full-eval baseline

## 9. Current Recommendation

Do not expand data scale yet.

The highest-value next step is:

- keep the current small baseline fixed
- continue reranker iteration on the small dev split
- target a stronger query-aware reranker while preserving the no-leakage workflow
