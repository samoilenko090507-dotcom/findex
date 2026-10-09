import time
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import tokenize
from findex.index import DocMeta, Index, Posting, open_index
from findex.parser import And, Not, Or, Term, parse_query
from findex.scoring import BM25Scorer, TfIdfScorer

runner = CliRunner()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Hello World", ["hello", "world"]),
        ("Привіт, світе!", ["привіт", "світе"]),
        ("café naïve 123", ["café", "naïve", "123"]),
        ("punctuation... and spaces!?", ["punctuation", "and", "spaces"]),
        ("", []),
    ],
)
def test_tokenizer_unicode_and_symbols(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


def test_index_build_counts(indexed_small: Index) -> None:
    assert indexed_small.num_docs == 3
    assert "apple" in indexed_small
    assert indexed_small.df("apple") == 1
    assert indexed_small.df("cherry") == 2


def test_index_mapping_protocol(indexed_small: Index) -> None:
    assert len(indexed_small) > 0
    assert list(indexed_small["apple"])
    with pytest.raises(KeyError):
        _ = indexed_small["nonexistent_term_xyz"]


def test_index_positions(indexed_small: Index) -> None:
    postings = indexed_small["cherry"]
    doc0_post = next(p for p in postings if p.doc_id == 0)
    assert doc0_post.positions == (3,)


def test_index_merge(small_corpus: Path) -> None:
    idx1 = Index()
    idx1.build_from_directory(small_corpus)
    idx2 = Index()
    idx2.build_from_directory(small_corpus)
    merged = idx1.merge(idx2)
    assert "apple" in merged
    assert merged["apple"][0].frequency == idx1["apple"][0].frequency * 2


def test_parser_single_term() -> None:
    tree = parse_query("apple")
    assert isinstance(tree, Term)
    assert tree.value == "apple"


def test_parser_and_node() -> None:
    tree = parse_query("apple AND banana")
    assert isinstance(tree, And)
    assert isinstance(tree.left, Term)
    assert isinstance(tree.right, Term)
    assert tree.left.value == "apple"
    assert tree.right.value == "banana"


def test_parser_or_node() -> None:
    tree = parse_query("apple OR banana")
    assert isinstance(tree, Or)
    assert tree.left.value == "apple"
    assert tree.right.value == "banana"


def test_parser_not_node() -> None:
    tree = parse_query("NOT apple")
    assert isinstance(tree, Not)
    assert isinstance(tree.child, Term)
    assert tree.child.value == "apple"


def test_parser_operator_precedence() -> None:
    tree = parse_query("apple OR banana AND cherry")
    assert isinstance(tree, Or)
    assert isinstance(tree.right, And)


def test_parser_syntax_error() -> None:
    tree = parse_query(")")
    assert isinstance(tree, Term)
    assert tree.value == ""


def test_bm25_rare_term_higher() -> None:
    scorer = BM25Scorer()
    rare = scorer.score_term(tf=1, doc_len=100, df=1, total_docs=100, avg_doc_len=100.0)
    common = scorer.score_term(
        tf=1, doc_len=100, df=80, total_docs=100, avg_doc_len=100.0
    )
    assert rare > common


def test_bm25_term_saturation() -> None:
    scorer = BM25Scorer()
    s1 = scorer.score_term(tf=1, doc_len=100, df=10, total_docs=100, avg_doc_len=100.0)
    s2 = scorer.score_term(tf=2, doc_len=100, df=10, total_docs=100, avg_doc_len=100.0)
    s10 = scorer.score_term(
        tf=10, doc_len=100, df=10, total_docs=100, avg_doc_len=100.0
    )
    s11 = scorer.score_term(
        tf=11, doc_len=100, df=10, total_docs=100, avg_doc_len=100.0
    )
    assert (s2 - s1) > (s11 - s10)


def test_bm25_length_normalization() -> None:
    scorer = BM25Scorer()
    short_doc = scorer.score_term(
        tf=2, doc_len=50, df=10, total_docs=100, avg_doc_len=100.0
    )
    long_doc = scorer.score_term(
        tf=2, doc_len=200, df=10, total_docs=100, avg_doc_len=100.0
    )
    assert short_doc > long_doc


def test_tfidf_scoring() -> None:
    scorer = TfIdfScorer()
    score = scorer.score_term(tf=2, doc_len=10, df=5, total_docs=50, avg_doc_len=10.0)
    assert score > 0.0


def test_binary_save_load(indexed_small: Index, tmp_path: Path) -> None:
    file_bin = tmp_path / "test.bin"
    indexed_small.save_binary(file_bin)
    loaded = Index.load_binary(file_bin)
    assert len(loaded) == len(indexed_small)
    assert loaded.num_docs == indexed_small.num_docs
    assert loaded["banana"] == indexed_small["banana"]


def test_pickle_save_load(indexed_small: Index, tmp_path: Path) -> None:
    file_pkl = tmp_path / "test.pkl"
    indexed_small.save_pickle(file_pkl)
    loaded = Index.load_pickle(file_pkl)
    assert len(loaded) == len(indexed_small)
    assert loaded.num_docs == indexed_small.num_docs


def test_context_manager_cleanup(indexed_small: Index, tmp_path: Path) -> None:
    file_bin = tmp_path / "idx.bin"
    indexed_small.save_binary(file_bin)
    with open_index(file_bin) as idx:
        assert len(idx) > 0
    assert len(idx) == 0


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "findex" in result.stdout


def test_cli_stats(indexed_small: Index, tmp_path: Path) -> None:
    bin_path = tmp_path / "test.bin"
    indexed_small.save_binary(bin_path)
    result = runner.invoke(app, ["stats", str(bin_path)])
    assert result.exit_code == 0
    assert "Статистика індексу" in result.stdout


def test_cli_search_json(
    indexed_small: Index, tmp_path: Path, small_corpus: Path
) -> None:
    bin_path = tmp_path / "test.bin"
    indexed_small.save_binary(bin_path)
    result = runner.invoke(
        app, ["search", str(bin_path), "apple", "--json", "--corpus", str(small_corpus)]
    )
    assert result.exit_code == 0
    assert '"score"' in result.stdout


def test_cli_error_missing_file() -> None:
    result = runner.invoke(app, ["search", "nonexistent_file_abc.bin", "apple"])
    assert result.exit_code != 0
    assert "Помилка" in result.stderr or "Помилка" in result.stdout


@pytest.mark.slow
def test_slow_dummy_sleep() -> None:
    time.sleep(1.05)
    assert True


@settings(max_examples=25)
@given(
    st.lists(
        st.text(alphabet=st.characters(categories=["L"]), min_size=1, max_size=8),
        min_size=1,
        max_size=10,
    )
)
def test_hypothesis_postings_always_sorted(words: list[str]) -> None:
    idx = Index()
    for doc_id, w in enumerate(words):
        t = w.lower()
        idx._postings[t].append(Posting(doc_id=doc_id, frequency=1))
    for post_list in idx._postings.values():
        doc_ids = [p.doc_id for p in post_list]
        assert doc_ids == sorted(doc_ids)


@settings(max_examples=25)
@given(st.lists(st.integers(min_value=0, max_value=50), min_size=1, max_size=10))
def test_hypothesis_merge_matches_set_union(doc_ids: list[int]) -> None:
    idx1 = Index()
    idx2 = Index()
    half = len(doc_ids) // 2
    for d_id in doc_ids[:half]:
        idx1._postings["test"].append(Posting(doc_id=d_id, frequency=1))
    for d_id in doc_ids[half:]:
        idx2._postings["test"].append(Posting(doc_id=d_id, frequency=1))
    merged = idx1.merge(idx2)
    expected_ids = {p.doc_id for p in idx1._postings.get("test", [])} | {
        p.doc_id for p in idx2._postings.get("test", [])
    }
    actual_ids = {p.doc_id for p in merged._postings.get("test", [])}
    assert actual_ids == expected_ids


def test_hypothesis_save_load_equivalence(tmp_path: Path) -> None:
    @settings(max_examples=15)
    @given(
        st.lists(
            st.text(alphabet="abcdef", min_size=2, max_size=5), min_size=1, max_size=5
        )
    )
    def run_property(terms: list[str]) -> None:
        idx = Index()
        for i, t in enumerate(terms):
            idx._postings[t.lower()].append(
                Posting(doc_id=i, frequency=1, positions=(0,))
            )
            idx.documents[i] = DocMeta(doc_id=i, name=f"doc{i}.txt", length=1)
        file_bin = tmp_path / "hypo.bin"
        idx.save_binary(file_bin)
        loaded = Index.load_binary(file_bin)
        assert set(loaded.keys()) == set(idx.keys())
        for t in idx:
            assert loaded[t] == idx[t]

    run_property()
