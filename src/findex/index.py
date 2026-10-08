import argparse
import array
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, lru_cache, wraps
import os
from pathlib import Path
import pickle
import struct
import sys
import time
import tracemalloc
from typing import Dict, Iterator, List, Optional, Tuple

from findex.corpus import iter_documents, tokenize


def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"[@timed] {func.__name__} виконано за {elapsed:.5f} с")
        return result
    return wrapper


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    frequency: int
    positions: Tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    name: str
    length: int


class Index(Mapping):
    def __init__(self):
        self._postings: Dict[str, List[Posting]] = defaultdict(list)
        self.documents: Dict[int, DocMeta] = {}

    def __len__(self) -> int:
        return len(self._postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __getitem__(self, term: str) -> List[Posting]:
        return self._postings[term]

    def __contains__(self, term: object) -> bool:
        return term in self._postings

    def __repr__(self) -> str:
        return f"<Index terms={len(self)} docs={self.num_docs}>"

    @property
    def num_docs(self) -> int:
        return len(self.documents)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self.documents:
            return 0.0
        total_len = sum(doc.length for doc in self.documents.values())
        return total_len / len(self.documents)

    def df(self, term: str) -> int:
        term_norm = term.lower()
        if term_norm in self._postings:
            return len(self._postings[term_norm])
        return 0

    @timed
    def build_from_directory(self, corpus_path: Path, store_positions: bool = True):
        for doc_id, (file_path, text) in enumerate(iter_documents(corpus_path)):
            tokens = list(tokenize(text))
            doc_len = len(tokens)
            p_obj = Path(file_path)
            self.documents[doc_id] = DocMeta(
                doc_id=doc_id,
                name=p_obj.name,
                length=doc_len
            )
            term_counts = Counter(tokens)
            positions_map = defaultdict(list)
            if store_positions:
                for idx, t in enumerate(tokens):
                    positions_map[t].append(idx)

            for term, count in term_counts.items():
                pos = tuple(positions_map[term]) if store_positions else ()
                self._postings[term].append(
                    Posting(doc_id=doc_id, frequency=count, positions=pos)
                )

        for term in self._postings:
            self._postings[term].sort(key=lambda p: p.doc_id)

    @timed
    def save_pickle(self, filepath: Path):
        with open(filepath, "wb") as f:
            pickle.dump({"documents": self.documents, "postings": dict(self._postings)}, f)

    @classmethod
    @timed
    def load_pickle(cls, filepath: Path) -> "Index":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx.documents = data["documents"]
        idx._postings = defaultdict(list, data["postings"])
        return idx

    @timed
    def save_binary(self, filepath: Path):
        with open(filepath, "wb") as f:
            f.write(struct.pack("<I", len(self.documents)))
            for doc in self.documents.values():
                name_bytes = doc.name.encode("utf-8")
                f.write(struct.pack("<II", doc.doc_id, len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", doc.length))

            f.write(struct.pack("<I", len(self._postings)))
            for term, post_list in self._postings.items():
                term_bytes = term.encode("utf-8")
                f.write(struct.pack("<I", len(term_bytes)))
                f.write(term_bytes)
                n = len(post_list)
                f.write(struct.pack("<I", n))
                for p in post_list:
                    pos_len = len(p.positions)
                    f.write(struct.pack("<III", p.doc_id, p.frequency, pos_len))
                    if pos_len > 0:
                        pos_arr = array.array("I", p.positions)
                        pos_arr.tofile(f)

    @classmethod
    @timed
    def load_binary(cls, filepath: Path) -> "Index":
        idx = cls()
        with open(filepath, "rb") as f:
            num_docs = struct.unpack("<I", f.read(4))[0]
            for _ in range(num_docs):
                doc_id, name_len = struct.unpack("<II", f.read(8))
                name = f.read(name_len).decode("utf-8")
                length = struct.unpack("<I", f.read(4))[0]
                idx.documents[doc_id] = DocMeta(doc_id=doc_id, name=name, length=length)

            num_terms = struct.unpack("<I", f.read(4))[0]
            for _ in range(num_terms):
                term_len = struct.unpack("<I", f.read(4))[0]
                term = f.read(term_len).decode("utf-8")
                n = struct.unpack("<I", f.read(4))[0]
                postings = []
                for _ in range(n):
                    doc_id, freq, pos_len = struct.unpack("<III", f.read(12))
                    if pos_len > 0:
                        pos_arr = array.array("I")
                        pos_arr.fromfile(f, pos_len)
                        pos_tuple = tuple(pos_arr)
                    else:
                        pos_tuple = ()
                    postings.append(Posting(doc_id=doc_id, frequency=freq, positions=pos_tuple))
                idx._postings[term] = postings
        return idx


@contextmanager
def open_index(path: Path):
    path_obj = Path(path)
    if str(path_obj).endswith(".pkl"):
        index = Index.load_pickle(path_obj)
    else:
        index = Index.load_binary(path_obj)
    try:
        yield index
    finally:
        index.documents.clear()
        index._postings.clear()


@lru_cache(maxsize=128)
def cached_query_ids(query: str, index_path_str: str) -> Tuple[int, ...]:
    from findex.parser import parse_query
    with open_index(Path(index_path_str)) as index:
        tree = parse_query(query)
        doc_ids = tree.evaluate(index)
        return tuple(sorted(doc_ids))