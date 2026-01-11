#!/usr/bin/env python3
"""
Simple hybrid search:
- Semantic search (Sentence-Transformers embeddings + cosine similarity)
- BM25 (rank-bm25)
- Rank fusion with RRF (Reciprocal Rank Fusion)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ----------------------------
# Utilities
# ----------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)


def cosine_sim_matrix(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Returns cosine similarities between query_vec and each row in doc_vecs.
    Assumes both are already L2-normalized.
    """
    return doc_vecs @ query_vec


def argsort_desc(scores: np.ndarray) -> List[int]:
    return list(np.argsort(-scores))


# ----------------------------
# RRF (Reciprocal Rank Fusion)
# ----------------------------

def rrf_fusion(
    ranked_lists: List[List[int]],
    k: int = 60,
    weights: List[float] | None = None,
) -> Dict[int, float]:
    """
    ranked_lists: each list is doc_ids ordered best->worst
    k: RRF constant (common choices: 10, 60)
    weights: optional per-list weights, same length as ranked_lists
    Returns: dict doc_id -> fused_score
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    scores: Dict[int, float] = {}
    for w, rlist in zip(weights, ranked_lists):
        for rank, doc_id in enumerate(rlist, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (k + rank))
    return scores


# ----------------------------
# Data model
# ----------------------------

@dataclass
class Doc:
    id: int
    text: str


# ----------------------------
# Hybrid Search Engine
# ----------------------------

class HybridSearchEngine:
    def __init__(self, docs: List[Doc], model_name: str = "all-MiniLM-L6-v2"):
        self.docs = docs

        # BM25 index
        self.tokenized_docs = [tokenize(d.text) for d in docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Semantic index
        self.embedder = SentenceTransformer(model_name)
        self.doc_embeddings = self.embedder.encode(
            [d.text for d in docs],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        ).astype(np.float32)
        self.doc_embeddings = l2_normalize(self.doc_embeddings)

    def search(
        self,
        query: str,
        topk: int = 5,
        rrf_k: int = 60,
        semantic_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> Dict[str, List[Tuple[int, float]]]:
        # ----- BM25 -----
        q_tokens = tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)
        bm25_rank = argsort_desc(bm25_scores)

        # ----- Semantic -----
        q_emb = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )[0].astype(np.float32)
        q_emb = l2_normalize(q_emb)
        sem_scores = cosine_sim_matrix(q_emb, self.doc_embeddings)
        sem_rank = argsort_desc(sem_scores)

        # ----- RRF Fusion -----
        fused = rrf_fusion(
            ranked_lists=[sem_rank, bm25_rank],
            k=rrf_k,
            weights=[semantic_weight, bm25_weight],
        )
        fused_rank = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # Helper to pack results
        def pack(ranked_ids: List[int], scores: np.ndarray) -> List[Tuple[int, float]]:
            out = []
            for doc_id in ranked_ids[:topk]:
                out.append((doc_id, float(scores[doc_id])))
            return out

        return {
            "semantic_top": pack(sem_rank, sem_scores),
            "bm25_top": pack(bm25_rank, bm25_scores),
            "rrf_top": [(doc_id, float(score)) for doc_id, score in fused_rank[:topk]],
        }


# ----------------------------
# Demo
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Hybrid Search: Semantic + BM25 + RRF")
    parser.add_argument("--query", required=True, help="Search query text")
    parser.add_argument("--topk", type=int, default=5, help="How many results to show")
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF constant (k)")
    parser.add_argument("--semantic_weight", type=float, default=1.0, help="Weight for semantic rank list")
    parser.add_argument("--bm25_weight", type=float, default=1.0, help="Weight for BM25 rank list")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-Transformers model name")
    args = parser.parse_args()

    # Example mini corpus
    docs = [
        Doc(0, "Our refund policy allows returns within 14 days for unused subscriptions."),
        Doc(1, "You can reset your password using the account settings page."),
        Doc(2, "Subscriptions renew automatically every month unless you cancel before the renewal date."),
        Doc(3, "To request a refund, contact support with your invoice number and reason."),
        Doc(4, "We offer enterprise plans with SSO and advanced security controls."),
        Doc(5, "Cancel your plan anytime. Access remains until the end of the billing period."),
        Doc(6, "Billing issues: failed payments can be resolved by updating your payment method."),
    ]

    engine = HybridSearchEngine(docs, model_name=args.model)
    results = engine.search(
        query=args.query,
        topk=args.topk,
        rrf_k=args.rrf_k,
        semantic_weight=args.semantic_weight,
        bm25_weight=args.bm25_weight,
    )

    def print_block(title: str, items: List[Tuple[int, float]]):
        print(f"\n=== {title} ===")
        for doc_id, score in items:
            print(f"[{doc_id}] score={score:.4f} | {docs[doc_id].text}")

    print_block("Semantic (cosine) Top", results["semantic_top"])
    print_block("BM25 Top", results["bm25_top"])
    print_block("RRF Fused Top", results["rrf_top"])


if __name__ == "__main__":
    main()
