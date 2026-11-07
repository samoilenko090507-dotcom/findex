from __future__ import annotations
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

class FastScorer:
    @staticmethod
    def bm25(
        doc_ids: np.ndarray,
        tfs: np.ndarray,
        doc_lens: np.ndarray,
        avgdl: float,
        df: int,
        n_docs: int,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> np.ndarray:
        if len(doc_ids) == 0:
            return np.array([], dtype=np.float32)
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        lens = doc_lens[doc_ids]
        num = tfs * (k1 + 1.0)
        den = tfs + k1 * (1.0 - b + b * (lens / avgdl))
        return (idf * (num / den)).astype(np.float32)

    @staticmethod
    def top_k(scores: np.ndarray, doc_ids: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        if len(scores) == 0:
            return []
        k = min(k, len(scores))
        part_idx = np.argpartition(-scores, k - 1)[:k]
        sorted_top = part_idx[np.argsort(-scores[part_idx])]
        return [(int(doc_ids[i]), float(scores[i])) for i in sorted_top]

class SemanticEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_embeddings(self, texts: List[str], save_path: Path) -> np.ndarray:
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        np.save(save_path, emb.astype(np.float32))
        return emb

    @staticmethod
    def search(query_vector: np.ndarray, doc_embeddings: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        scores = np.dot(doc_embeddings, query_vector)
        k = min(k, len(scores))
        part_idx = np.argpartition(-scores, k - 1)[:k]
        sorted_top = part_idx[np.argsort(-scores[part_idx])]
        return [(int(i), float(scores[i])) for i in sorted_top]

def rrf(bm25_results: List[int], sem_results: List[int], k: int = 60, top_n: int = 10) -> List[int]:
    scores: Dict[int, float] = {}
    for rank, doc_id in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(sem_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:top_n]]