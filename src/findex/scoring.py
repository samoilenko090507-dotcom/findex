import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SearchResult:
    score: float
    doc_id: int

    def __lt__(self, other: "SearchResult") -> bool:
        return self.score < other.score


@runtime_checkable
class Scorer(Protocol):
    def score_term(
        self,
        tf: int,
        doc_len: int,
        df: int,
        total_docs: int,
        avg_doc_len: float,
    ) -> float: ...


class TfIdfScorer:
    def score_term(
        self,
        tf: int,
        doc_len: int,
        df: int,
        total_docs: int,
        avg_doc_len: float,
    ) -> float:
        if tf <= 0 or df <= 0 or total_docs <= 0:
            return 0.0
        term_freq = tf / max(1, doc_len)
        idf = math.log((total_docs + 1) / (df + 1)) + 1.0
        return term_freq * idf


class BM25Scorer:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def score_term(
        self,
        tf: int,
        doc_len: int,
        df: int,
        total_docs: int,
        avg_doc_len: float,
    ) -> float:
        if tf <= 0 or df <= 0 or total_docs <= 0:
            return 0.0
        numerator = total_docs - df + 0.5
        denominator = df + 0.5
        idf = math.log(1.0 + (numerator / denominator))
        avg_len = avg_doc_len if avg_doc_len > 0 else 1.0
        len_norm = 1.0 - self.b + self.b * (doc_len / avg_len)
        tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
        return idf * tf_component
