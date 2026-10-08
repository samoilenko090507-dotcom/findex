import argparse
import heapq
from pathlib import Path
from typing import List, Tuple

from findex.corpus import iter_documents, tokenize
from findex.index import Index, open_index
from findex.parser import parse_query
from findex.scoring import BM25Scorer, Scorer, SearchResult, TfIdfScorer


def extract_snippet(text: str, query_tokens: List[str], window: int = 80) -> str:
    tokens = list(tokenize(text))
    if not tokens:
        return text[: window * 2]

    lower_query = {t.lower() for t in query_tokens}
    best_idx = -1
    for idx, t in enumerate(tokens):
        if t in lower_query:
            best_idx = idx
            break

    if best_idx == -1:
        return text[: window * 2].strip() + "..."

    target_word = tokens[best_idx]
    pos = text.lower().find(target_word)
    if pos == -1:
        return text[: window * 2].strip() + "..."

    start = max(0, pos - window)
    end = min(len(text), pos + len(target_word) + window)
    raw_snippet = text[start:end].replace("\n", " ").strip()

    highlighted = raw_snippet
    for q in lower_query:
        import re
        highlighted = re.sub(
            rf"\b({re.escape(q)})\b",
            r"[\1]",
            highlighted,
            flags=re.IGNORECASE,
        )

    return f"...{highlighted}..."


def ranked_search(
    index: Index,
    query_str: str,
    scorer: Scorer,
    corpus_dir: Path,
    top_k: int = 5,
) -> List[Tuple[float, int, str, str]]:
    tree = parse_query(query_str)
    candidate_doc_ids = tree.evaluate(index)
    if not candidate_doc_ids:
        return []

    query_terms = [t.lower() for t in tokenize(query_str) if t.lower() in index]
    if not query_terms:
        query_terms = [t.lower() for t in tokenize(query_str)]

    doc_texts = {}
    if corpus_dir.exists():
        for doc_id, (_, text) in enumerate(iter_documents(corpus_dir)):
            if doc_id in candidate_doc_ids:
                doc_texts[doc_id] = text

    heap: List[SearchResult] = []
    total_docs = index.num_docs
    avg_len = index.avg_doc_length

    for doc_id in candidate_doc_ids:
        meta = index.documents.get(doc_id)
        doc_len = meta.length if meta else 1
        doc_score = 0.0

        for term in query_terms:
            if term in index:
                postings = index[term]
                tf = 0
                for p in postings:
                    if p.doc_id == doc_id:
                        tf = p.frequency
                        break
                df_val = index.df(term)
                term_score = scorer.score_term(tf, doc_len, df_val, total_docs, avg_len)
                doc_score += term_score

        res = SearchResult(score=doc_score, doc_id=doc_id)
        if len(heap) < top_k:
            heapq.heappush(heap, res)
        else:
            heapq.heappushpop(heap, res)

    top_results = heapq.nlargest(top_k, heap)
    results = []
    for item in top_results:
        meta = index.documents.get(item.doc_id)
        name = meta.name if meta else f"doc_{item.doc_id}"
        raw_text = doc_texts.get(item.doc_id, "")
        snippet = extract_snippet(raw_text, query_terms)
        results.append((item.score, item.doc_id, name, snippet))

    return results


def main():
    parser = argparse.ArgumentParser(description="Ранжований пошук findex (Лаб 3)")
    parser.add_argument("index_path", type=Path, help="Шлях до індексу (.bin або .pkl)")
    parser.add_argument("query", type=str, help="Пошуковий запит")
    parser.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25", help="Модель ранжування")
    parser.add_argument("--corpus", type=Path, default=Path("data"), help="Шлях до теки корпусу для сніпетів")
    parser.add_argument("--top", type=int, default=5, help="Кількість результатів")
    args = parser.parse_args()

    scorer: Scorer = BM25Scorer() if args.scorer == "bm25" else TfIdfScorer()

    with open_index(args.index_path) as index:
        results = ranked_search(index, args.query, scorer, args.corpus, top_k=args.top)

    print(f"\nРезультати пошуку ({args.scorer.upper()}, top-{args.top}):")
    if not results:
        print("Нічого не знайдено.")
        return

    for rank, (score, doc_id, name, snippet) in enumerate(results, start=1):
        print(f"\n{rank}. [{name}] (Doc ID: {doc_id}, Score: {score:.4f})")
        print(f"   Сніпет: {snippet}")


if __name__ == "__main__":
    main()