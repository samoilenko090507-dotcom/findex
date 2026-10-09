import array
import os
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents, tokenize
from findex.index import InvertedIndex
from findex.search import boolean_search_merge, boolean_search_set


@dataclass(frozen=True)
class PostingNoSlots:
    doc_id: int
    frequency: int


@dataclass(frozen=True, slots=True)
class PostingSlots:
    doc_id: int
    frequency: int


def benchmark_storage(raw_data: list[tuple[int, list[str]]]):
    results = {}

    tracemalloc.start()
    idx_no_slots = defaultdict(list)
    for doc_id, tokens in raw_data:
        counts = Counter(tokens)
        for term, c in counts.items():
            idx_no_slots[term].append(PostingNoSlots(doc_id=doc_id, frequency=c))
    _, peak_no_slots = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["no_slots"] = peak_no_slots / (1024 * 1024)
    del idx_no_slots

    tracemalloc.start()
    idx_slots = defaultdict(list)
    for doc_id, tokens in raw_data:
        counts = Counter(tokens)
        for term, c in counts.items():
            idx_slots[term].append(PostingSlots(doc_id=doc_id, frequency=c))
    _, peak_slots = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["slots"] = peak_slots / (1024 * 1024)
    del idx_slots

    tracemalloc.start()
    idx_arrays = defaultdict(lambda: (array.array("I"), array.array("I")))
    for doc_id, tokens in raw_data:
        counts = Counter(tokens)
        for term, c in counts.items():
            idx_arrays[term][0].append(doc_id)
            idx_arrays[term][1].append(c)
    _, peak_arrays = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["arrays"] = peak_arrays / (1024 * 1024)
    del idx_arrays

    return results


def run_all_benchmarks():
    corpus_dir = Path("data")
    if not corpus_dir.exists():
        corpus_dir = Path("src/findex/data")

    raw_data = []
    for doc_id, (_, text) in enumerate(iter_documents(corpus_dir)):
        raw_data.append((doc_id, list(tokenize(text))))

    storage_res = benchmark_storage(raw_data)

    index = InvertedIndex()
    index.build_from_directory(corpus_dir)

    pickle_path = Path("index.pkl")
    binary_path = Path("index.bin")

    t0 = time.perf_counter()
    index.save_pickle(pickle_path)
    t_save_pkl = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = InvertedIndex.load_pickle(pickle_path)
    t_load_pkl = time.perf_counter() - t0
    size_pkl = os.path.getsize(pickle_path) / 1024

    t0 = time.perf_counter()
    index.save_binary(binary_path)
    t_save_bin = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = InvertedIndex.load_binary(binary_path)
    t_load_bin = time.perf_counter() - t0
    size_bin = os.path.getsize(binary_path) / 1024

    sorted_terms = sorted(
        index.postings.keys(), key=lambda t: len(index.postings[t]), reverse=True
    )
    frequent_terms = f"{sorted_terms[0]} {sorted_terms[1]}"
    rare_terms = f"{sorted_terms[-1]} {sorted_terms[-2]}"

    def test_search(query: str, runs=500):
        t0 = time.perf_counter()
        for _ in range(runs):
            boolean_search_merge(index, query)
        t_merge = (time.perf_counter() - t0) / runs

        t0 = time.perf_counter()
        for _ in range(runs):
            boolean_search_set(index, query)
        t_set = (time.perf_counter() - t0) / runs
        return t_merge * 1e6, t_set * 1e6

    freq_merge_us, freq_set_us = test_search(frequent_terms)
    rare_merge_us, rare_set_us = test_search(rare_terms)

    print("=== РЕЗУЛЬТАТИ ВИМІРЮВАНЬ ===")
    print(
        f"Пам'ять: no_slots={storage_res['no_slots']:.2f} МБ | slots={storage_res['slots']:.2f} МБ | arrays={storage_res['arrays']:.2f} МБ"
    )
    print(
        f"Pickle: розмір={size_pkl:.2f} КБ | save={t_save_pkl:.4f} с | load={t_load_pkl:.4f} с"
    )
    print(
        f"Binary: розмір={size_bin:.2f} КБ | save={t_save_bin:.4f} с | load={t_load_bin:.4f} с"
    )
    print(
        f"Пошук (найчастіші: '{frequent_terms}'): merge={freq_merge_us:.2f} мкс | set={freq_set_us:.2f} мкс"
    )
    print(
        f"Пошук (найрідкісніші: '{rare_terms}'): merge={rare_merge_us:.2f} мкс | set={rare_set_us:.2f} мкс"
    )


if __name__ == "__main__":
    run_all_benchmarks()
