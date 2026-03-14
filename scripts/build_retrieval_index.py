import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src_ext.retrieval.candidate_builder import build_train_candidates, load_sequence_table
from src_ext.retrieval.retriever import SequenceRetriever
from src_ext.utils.config import load_config
from src_ext.utils.device import get_device
from src_ext.utils.paths import ensure_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    train_file = cfg["retrieval"].get("train_file", cfg["data"]["train_file"])
    train_df = load_sequence_table(cfg["project"]["root"] / train_file)
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
        retriever.fit_clean_precomputed_corpus(train_df, embedding_file)
    else:
        retriever.fit(
            build_train_candidates(train_df),
            dataset_name=f"{cfg['experiment']['name']}_train_retrieval",
        )
    out_path = cfg["paths"]["cache_dir"] / f"{cfg['experiment']['name']}_retriever.pkl"
    retriever.save(out_path)

    print("retrieval index saved to:", out_path)


if __name__ == "__main__":
    main()
