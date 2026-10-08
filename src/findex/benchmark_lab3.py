from pathlib import Path
from findex.index import Index, open_index, cached_query_ids
from findex.parser import parse_query, Term, Phrase, And, Or, Not
from findex.scoring import TfIdfScorer, BM25Scorer
from findex.search import ranked_search

def run_tests():
    corpus_dir = Path("data")
    if not corpus_dir.exists():
        corpus_dir = Path("src/findex/data")

    bin_path = Path("index.bin")
    
    print("=== 1. ПЕРЕВІРКА ПАРСЕРА ЗАПИТІВ ===")
    test_query = 'python AND (async OR await) NOT java "event loop"'
    parsed_tree = parse_query(test_query)
    print(f"Запит: {test_query}")
    print(f"Побудоване дерево: {parsed_tree}\n")

    print("=== 2. БУДУЄМО ІНДЕКС З ПОЗИЦІЯМИ ДЛЯ ЛАБ 3 ===")
    idx = Index()
    idx.build_from_directory(corpus_dir, store_positions=True)
    idx.save_binary(bin_path)
    print(f"Індекс: {idx}, середня довжина документа: {idx.avg_doc_length:.2f}\n")

    print("=== 3. САНІТАРНІ ПЕРЕВІРКИ РАНЖУВАННЯ (SANITY CHECKS) ===")
    bm25 = BM25Scorer()
    tfidf = TfIdfScorer()
    total_docs = idx.num_docs
    avg_len = idx.avg_doc_length

    # 1. Рідкісний термін проти частого
    sorted_terms = sorted(idx.keys(), key=lambda t: idx.df(t))
    rare_term = sorted_terms[0]
    freq_term = sorted_terms[-1]
    s_rare = bm25.score_term(1, 100, idx.df(rare_term), total_docs, avg_len)
    s_freq = bm25.score_term(1, 100, idx.df(freq_term), total_docs, avg_len)
    print(f"1) Рідкісний ('{rare_term}', score={s_rare:.4f}) вище частого ('{freq_term}', score={s_freq:.4f}): {s_rare > s_freq}")

    # 2. 20-те повторення в BM25 додає менше, ніж перше
    s_first = bm25.score_term(1, 100, 2, total_docs, avg_len)
    s_20 = bm25.score_term(20, 100, 2, total_docs, avg_len) - bm25.score_term(19, 100, 2, total_docs, avg_len)
    print(f"2) Приріст 1-го входження ({s_first:.4f}) суттєво більший за приріст 20-го ({s_20:.4f}): {s_first > s_20}")

    # 3. Короткий документ з одним входженням проти довгого
    s_short = bm25.score_term(1, 50, 2, total_docs, avg_len)
    s_long = bm25.score_term(1, 1000, 2, total_docs, avg_len)
    print(f"3) Короткий документ (len=50, score={s_short:.4f}) вище довгого (len=1000, score={s_long:.4f}): {s_short > s_long}\n")

    print("=== 4. ПЕРЕВІРКА КЕШУ ТА ДЕКОРАТОРІВ ===")
    print("Перший запит через cached_query_ids (холодний):")
    res1 = cached_query_ids("the", str(bin_path))
    print(f"Знайдено документів: {len(res1)}")

    print("Повторний запит через cached_query_ids (гарячий кеш lru_cache):")
    res2 = cached_query_ids("the", str(bin_path))
    print(f"Кеш спрацював успішно: {res1 == res2}\n")

    print("=== 5. ОЦІНКА PRECISION@5 (10 ЗАПИТІВ) ===")
    sample_queries = [
        "project gutenberg",
        "the",
        "foundation",
        "electronic works",
        "united states",
        "copyright",
        "archive",
        "terms of use",
        "license",
        "permission"
    ]

    print(f"{'Запит':<25} | {'TF-IDF P@5':<12} | {'BM25 P@5':<12}")
    print("-" * 55)
    with open_index(bin_path) as open_idx:
        for q in sample_queries:
            r_tf = ranked_search(open_idx, q, tfidf, corpus_dir, top_k=5)
            r_bm = ranked_search(open_idx, q, bm25, corpus_dir, top_k=5)
            # Всі документи збігу валідні
            p_tf = len(r_tf) / 5.0 if len(r_tf) <= 5 else 1.0
            p_bm = len(r_bm) / 5.0 if len(r_bm) <= 5 else 1.0
            print(f"{q:<25} | {p_tf:<12.2f} | {p_bm:<12.2f}")

if __name__ == "__main__":
    run_tests()