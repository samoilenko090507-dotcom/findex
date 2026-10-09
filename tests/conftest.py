from pathlib import Path

import pytest

from findex.index import Index


@pytest.fixture
def small_corpus(tmp_path: Path) -> Path:
    d1 = tmp_path / "doc1.txt"
    d1.write_text("apple banana apple cherry", encoding="utf-8")
    d2 = tmp_path / "doc2.txt"
    d2.write_text("banana cherry date cherry", encoding="utf-8")
    d3 = tmp_path / "doc3.txt"
    d3.write_text("elderberry fig grape", encoding="utf-8")
    return tmp_path


@pytest.fixture
def indexed_small(small_corpus: Path) -> Index:
    idx = Index()
    idx.build_from_directory(small_corpus, store_positions=True)
    return idx
