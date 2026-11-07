import numpy as np
import pytest
from findex.fast_search import FastScorer, rrf

def test_bm25_matches_topk():
    doc_ids = np.array([0, 1, 2, 3], dtype=np.int32)
    tfs = np.array([5, 1, 3, 2], dtype=np.int32)
    doc_lens = np.array([100, 20, 50, 80], dtype=np.int32)
    scores = FastScorer.bm25(doc_ids, tfs, doc_lens, avgdl=50.0, df=4, n_docs=10)
    top = FastScorer.top_k(scores, doc_ids, k=2)
    assert len(top) == 2
    assert top[0][0] == 2

def test_rrf_merging():
    bm25_ranks = [10, 20, 30]
    sem_ranks = [20, 10, 40]
    merged = rrf(bm25_ranks, sem_ranks, top_n=2)
    assert 20 in merged

def test_benchmark_bm25_vectorized(benchmark):
    doc_ids = np.arange(10000, dtype=np.int32)
    tfs = np.random.randint(1, 10, size=10000, dtype=np.int32)
    doc_lens = np.random.randint(50, 500, size=10000, dtype=np.int32)
    
    def run_scoring():
        scores = FastScorer.bm25(doc_ids, tfs, doc_lens, avgdl=200.0, df=500, n_docs=20000)
        return FastScorer.top_k(scores, doc_ids, k=10)
        
    result = benchmark(run_scoring)
    assert len(result) == 10