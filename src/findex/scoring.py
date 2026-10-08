import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, order=True)
class SearchResult:
    score: float
    doc_id: int


@runtime_checkable
class Scorer(Protocol):
    def score_term(self, term_freq: int, doc_len: int, doc_freq: int, total_docs: int, avg_doc_len: float) -> float:
        ...


class TfIdfScorer:
    def score_term(self, term_freq: int, doc_len: int, doc_freq: int, total_docs: int, avg_doc_len: float) -> float:
        if term_freq <= 0 or doc_freq <= 0:
            return 0.0
        tf = term_freq / max(doc_len, 1)
        idf = math.log((total_docs + 1) / (doc_freq + 1)) + 1.0
        return tf * idf


class BM25Scorer:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score_term(self, term_freq: int, doc_len: int, doc_freq: int, total_docs: int, avg_doc_len: float) -> float:
        if term_freq <= 0 or doc_freq <= 0:
            return 0.0
        idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        numerator = term_freq * (self.k1 + 1.0)
        len_norm = 1.0 - self.b + self.b * (doc_len / max(avg_doc_len, 1.0))
        denominator = term_freq + self.k1 * len_norm
        return idf * (numerator / denominator)