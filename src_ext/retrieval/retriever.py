import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src_ext.retrieval.candidate_builder import build_clean_membership_table, split_ec_numbers
from src_ext.retrieval.faiss_index import SimpleVectorIndex


def _resolve_project_root(project_root=None):
    if project_root is not None:
        return Path(project_root).resolve()
    return Path(__file__).resolve().parents[2]


def _prepare_clean_imports(project_root):
    app_src = project_root / "app" / "src"
    if str(app_src) not in sys.path:
        sys.path.insert(0, str(app_src))

    from CLEAN.model import LayerNormNet

    return LayerNormNet


def _write_fasta(df, fasta_path):
    with open(fasta_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f">{row['Entry']}\n")
            f.write(f"{row['Sequence']}\n")


def _format_esm(payload):
    if isinstance(payload, dict):
        return payload["mean_representations"][33]
    return payload


class CleanEmbeddingRetriever:
    VERSION = 3

    def __init__(
        self,
        project_root=None,
        clean_train_data="split100",
        device="cpu",
        prototype_topk=5,
        neighbor_max_weight=0.40,
        neighbor_sum_weight=0.30,
        neighbor_count_weight=0.10,
        prototype_weight=0.20,
    ):
        self.project_root = _resolve_project_root(project_root)
        self.app_dir = self.project_root / "app"
        self.app_data_dir = self.app_dir / "data"
        self.esm_data_dir = self.app_data_dir / "esm_data"
        self.clean_train_data = clean_train_data
        self.device = torch.device(device)
        self.vector_index = SimpleVectorIndex()
        self.prototype_index = SimpleVectorIndex()
        self.train_df = None
        self.train_vectors = None
        self.ec_prototypes = {}
        self.prototype_topk = prototype_topk
        self.neighbor_max_weight = neighbor_max_weight
        self.neighbor_sum_weight = neighbor_sum_weight
        self.neighbor_count_weight = neighbor_count_weight
        self.prototype_weight = prototype_weight
        self.version = self.VERSION

    def _ensure_esm_embeddings(self, df, dataset_name):
        self.esm_data_dir.mkdir(parents=True, exist_ok=True)
        missing_mask = ~df["Entry"].map(lambda entry: (self.esm_data_dir / f"{entry}.pt").exists())
        if not missing_mask.any():
            return

        missing_df = df.loc[missing_mask, ["Entry", "Sequence"]].drop_duplicates().reset_index(drop=True)
        fasta_path = self.app_data_dir / f"{dataset_name}.fasta"
        _write_fasta(missing_df, fasta_path)

        cmd = [
            sys.executable,
            "esm/scripts/extract.py",
            "esm1b_t33_650M_UR50S",
            str(fasta_path.relative_to(self.app_dir)),
            str(self.esm_data_dir.relative_to(self.app_dir)),
            "--include",
            "mean",
        ]
        subprocess.run(cmd, check=True, cwd=self.app_dir)

    def _load_clean_model(self):
        LayerNormNet = _prepare_clean_imports(self.project_root)
        model = LayerNormNet(512, 128, self.device, torch.float32)
        checkpoint = torch.load(
            self.app_data_dir / "pretrained" / f"{self.clean_train_data}.pth",
            map_location=self.device,
        )
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def _load_esm_matrix(self, df):
        tensors = []
        for entry in df["Entry"].tolist():
            payload = torch.load(self.esm_data_dir / f"{entry}.pt", map_location="cpu")
            tensors.append(_format_esm(payload).unsqueeze(0))
        return torch.cat(tensors, dim=0).to(device=self.device, dtype=torch.float32)

    def encode(self, df, dataset_name):
        df = df.reset_index(drop=True).copy()
        self._ensure_esm_embeddings(df, dataset_name)
        model = self._load_clean_model()
        esm_matrix = self._load_esm_matrix(df)
        with torch.no_grad():
            vectors = model(esm_matrix).detach().cpu().numpy().astype(np.float32)
        return vectors

    def fit(self, train_df, dataset_name="clean_rag_train"):
        self.train_df = train_df.reset_index(drop=True).copy()
        self.train_vectors = self.encode(self.train_df, dataset_name=dataset_name)
        self.vector_index.build(self.train_vectors, self.train_df.index.tolist())
        self._build_ec_prototypes()
        return self

    def fit_from_precomputed(self, train_df, precomputed_vectors):
        train_df = train_df.reset_index(drop=True).copy()
        precomputed_vectors = np.asarray(precomputed_vectors, dtype=np.float32)
        if len(train_df) != len(precomputed_vectors):
            raise ValueError(
                f"precomputed vector count {len(precomputed_vectors)} does not match train rows {len(train_df)}"
            )

        self.train_df = train_df
        self.train_vectors = precomputed_vectors
        self.vector_index.build(self.train_vectors, self.train_df.index.tolist())
        self._build_ec_prototypes()
        return self

    def fit_clean_precomputed_corpus(self, train_df, embedding_path):
        membership_df = build_clean_membership_table(train_df)
        precomputed_vectors = (
            torch.load(embedding_path, map_location="cpu").detach().cpu().numpy().astype(np.float32)
        )
        return self.fit_from_precomputed(membership_df, precomputed_vectors)

    def _build_ec_prototypes(self):
        ec_to_vectors = {}
        ec_to_entries = {}
        for idx, row in self.train_df.iterrows():
            for ec in split_ec_numbers(row["EC number"]):
                ec_to_vectors.setdefault(ec, []).append(self.train_vectors[idx])
                ec_to_entries.setdefault(ec, []).append(row["Entry"])

        prototype_vectors = []
        prototype_ids = []
        self.ec_prototypes = {}
        for ec, vectors in ec_to_vectors.items():
            matrix = np.asarray(vectors, dtype=np.float32)
            prototype = matrix.mean(axis=0)
            prototype_vectors.append(prototype)
            prototype_ids.append(ec)
            self.ec_prototypes[ec] = {
                "vector": prototype,
                "support_entries": list(dict.fromkeys(ec_to_entries.get(ec, []))),
                "support_count": len(vectors),
            }

        if prototype_vectors:
            self.prototype_index.build(np.asarray(prototype_vectors, dtype=np.float32), prototype_ids)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def retrieve(self, test_df, topk=5, dataset_name="clean_rag_test"):
        if self.train_df is None or self.train_vectors is None:
            raise ValueError("retriever has not been fit")

        test_df = test_df.reset_index(drop=True).copy()
        query_vectors = self.encode(test_df, dataset_name=dataset_name)
        top_scores, top_ids = self.vector_index.search(query_vectors, topk=topk)
        prototype_scores, prototype_ids = self.prototype_index.search(
            query_vectors,
            topk=min(self.prototype_topk, max(1, len(self.ec_prototypes))),
        )

        results = {}
        for row_idx, (_, row) in enumerate(test_df.iterrows()):
            entry = row["Entry"]
            neighbors = []
            ec_scores = {}
            raw_neighbor_scores = [max(float(score), 0.0) for score in top_scores[row_idx]]
            retrieval_top1_raw = raw_neighbor_scores[0] if raw_neighbor_scores else 0.0
            retrieval_top2_raw = raw_neighbor_scores[1] if len(raw_neighbor_scores) > 1 else 0.0
            retrieval_margin_raw = retrieval_top1_raw - retrieval_top2_raw

            for score, train_idx in zip(top_scores[row_idx], top_ids[row_idx]):
                score = max(float(score), 0.0)
                train_row = self.train_df.iloc[train_idx]
                neighbor = {
                    "entry": train_row["Entry"],
                    "ec_numbers": split_ec_numbers(train_row["EC number"]),
                    "score": score,
                    "sequence": train_row["Sequence"],
                }
                neighbors.append(neighbor)

                for ec in neighbor["ec_numbers"]:
                    current = ec_scores.setdefault(
                        ec,
                        {
                            "neighbor_max_score": float("-inf"),
                            "neighbor_sum_score": 0.0,
                            "neighbor_count": 0,
                            "support_entries": [],
                            "prototype_score": 0.0,
                        },
                    )
                    current["neighbor_max_score"] = max(current["neighbor_max_score"], score)
                    current["neighbor_sum_score"] += score
                    current["neighbor_count"] += 1
                    if train_row["Entry"] not in current["support_entries"]:
                        current["support_entries"].append(train_row["Entry"])

            prototype_candidates = []
            for score, ec in zip(prototype_scores[row_idx], prototype_ids[row_idx]):
                score = max(float(score), 0.0)
                info = self.ec_prototypes.get(ec, {})
                prototype_candidates.append(
                    {
                        "ec_number": ec,
                        "score": score,
                        "support_count": int(info.get("support_count", 0)),
                        "support_entries": info.get("support_entries", []),
                    }
                )
                current = ec_scores.setdefault(
                    ec,
                    {
                        "neighbor_max_score": 0.0,
                        "neighbor_sum_score": 0.0,
                        "neighbor_count": 0,
                        "support_entries": [],
                        "prototype_score": 0.0,
                    },
                )
                current["prototype_score"] = max(current["prototype_score"], score)
                for support_entry in info.get("support_entries", []):
                    if support_entry not in current["support_entries"]:
                        current["support_entries"].append(support_entry)

            max_neighbor_max = max((payload["neighbor_max_score"] for payload in ec_scores.values()), default=1.0)
            max_neighbor_sum = max((payload["neighbor_sum_score"] for payload in ec_scores.values()), default=1.0)
            max_neighbor_count = max((payload["neighbor_count"] for payload in ec_scores.values()), default=1)
            max_prototype = max((payload["prototype_score"] for payload in ec_scores.values()), default=1.0)
            if max_neighbor_max <= 0:
                max_neighbor_max = 1.0
            if max_neighbor_sum <= 0:
                max_neighbor_sum = 1.0
            if max_neighbor_count <= 0:
                max_neighbor_count = 1
            if max_prototype <= 0:
                max_prototype = 1.0

            ranked_ecs = [
                {
                    "ec_number": ec,
                    "score": (
                        self.neighbor_max_weight * (payload["neighbor_max_score"] / max_neighbor_max)
                        + self.neighbor_sum_weight * (payload["neighbor_sum_score"] / max_neighbor_sum)
                        + self.neighbor_count_weight * (payload["neighbor_count"] / max_neighbor_count)
                        + self.prototype_weight * (payload["prototype_score"] / max_prototype)
                    ),
                    "neighbor_max_score": payload["neighbor_max_score"],
                    "neighbor_sum_score": payload["neighbor_sum_score"],
                    "neighbor_count": payload["neighbor_count"],
                    "prototype_score": payload["prototype_score"],
                    "support_entries": payload["support_entries"],
                }
                for ec, payload in sorted(
                    ec_scores.items(),
                    key=lambda item: (
                        item[1]["prototype_score"],
                        item[1]["neighbor_sum_score"],
                        item[1]["neighbor_max_score"],
                        item[1]["neighbor_count"],
                    ),
                    reverse=True,
                )
            ]
            ranked_ecs.sort(key=lambda item: item["score"], reverse=True)

            ec_top1 = ranked_ecs[0]["score"] if ranked_ecs else 0.0
            ec_top2 = ranked_ecs[1]["score"] if len(ranked_ecs) > 1 else 0.0

            results[entry] = {
                "neighbors": neighbors,
                "prototype_candidates": prototype_candidates,
                "ec_candidates": ranked_ecs,
                "retrieval_top1_raw": retrieval_top1_raw,
                "retrieval_top2_raw": retrieval_top2_raw,
                "retrieval_margin_raw": retrieval_margin_raw,
                "ec_top1_score": ec_top1,
                "ec_top2_score": ec_top2,
                "ec_margin_score": ec_top1 - ec_top2,
            }
        return results


SequenceRetriever = CleanEmbeddingRetriever
