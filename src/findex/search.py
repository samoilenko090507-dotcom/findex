import argparse
import sys
import time
import tracemalloc
from pathlib import Path
from typing import List, Set

from findex.corpus import tokenize
from findex.index import InvertedIndex, Posting


def intersect_postings(p1: List[Posting], p2: List[Posting]) -> List[Posting]:
    i, j = 0, 0
    result: List[Posting] = []
    while i < len(p1) and j < len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            result.append(p1[i])
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            i += 1
        else:
            j += 1
    return result


def boolean_search_merge(index: InvertedIndex, query: str) -> List[int]:
    raw_tokens = list(tokenize(query))
    if not raw_tokens:
        return []

    tokens: List[str] = []
    exclude_tokens: List[str] = []
    is_not = False

    for t in raw_tokens:
        if t in ("not", "ні"):
            is_not = True
            continue
        if is_not:
            exclude_tokens.append(t)
            is_not = False
        elif t not in ("and", "і", "та", "or", "або"):
            tokens.append(t)

    if not tokens:
        return []

    tokens.sort(key=lambda t: len(index.postings.get(t, [])))

    first_postings = index.postings.get(tokens[0], [])
    current_postings = list(first_postings)

    for term in tokens[1:]:
        term_postings = index.postings.get(term, [])
        current_postings = intersect_postings(current_postings, term_postings)
        if not current_postings:
            break

    result_doc_ids = [p.doc_id for p in current_postings]

    for ex in exclude_tokens:
        ex_ids = {p.doc_id for p in index.postings.get(ex, [])}
        result_doc_ids = [did for did in result_doc_ids if did not in ex_ids]

    return result_doc_ids


def boolean_search_set(index: InvertedIndex, query: str) -> List[int]:
    raw_tokens = list(tokenize(query))
    if not raw_tokens:
        return []

    tokens: List[str] = []
    exclude_tokens: List[str] = []
    is_not = False

    for t in raw_tokens:
        if t in ("not", "ні"):
            is_not = True
            continue
        if is_not:
            exclude_tokens.append(t)
            is_not = False
        elif t not in ("and", "і", "та", "or", "або"):
            tokens.append(t)

    if not tokens:
        return []

    result_set: Set[int] = {p.doc_id for p in index.postings.get(tokens[0], [])}

    for term in tokens[1:]:
        term_docs = {p.doc_id for p in index.postings.get(term, [])}
        result_set.intersection_update(term_docs)

    for ex in exclude_tokens:
        ex_docs = {p.doc_id for p in index.postings.get(ex, [])}
        result_set.difference_update(ex_docs)

    return sorted(result_set)


def main():
    parser = argparse.ArgumentParser(description="Пошук у двійковому індексі findex")
    parser.add_argument("index_path", type=Path, help="Шлях до збереженого файлу індексу")
    parser.add_argument("query", type=str, help="Пошуковий запит (наприклад, 'слово1 слово2')")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge", help="Алгоритм пошуку")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    if str(args.index_path).endswith(".pkl"):
        index = InvertedIndex.load_pickle(args.index_path)
    else:
        index = InvertedIndex.load_binary(args.index_path)

    if args.engine == "merge":
        doc_ids = boolean_search_merge(index, args.query)
    else:
        doc_ids = boolean_search_set(index, args.query)

    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Знайдено документів: {len(doc_ids)} (рушій: {args.engine})")
    for doc_id in doc_ids:
        meta = index.documents.get(doc_id)
        if meta:
            print(f"  [ID {meta.doc_id}] {meta.name} (довжина: {meta.length} токенів)")

    print(f"Час виконання: {elapsed_time:.6f} с")
    print(f"Пікова пам'ять: {peak_memory / 1024:.2f} КБ")


if __name__ == "__main__":
    main()