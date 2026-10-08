import argparse
import array
import os
import pickle
import struct
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from findex.corpus import iter_documents, tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    frequency: int


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    name: str
    length: int


class InvertedIndex:
    def __init__(self):
        self.postings: Dict[str, List[Posting]] = defaultdict(list)
        self.documents: Dict[int, DocMeta] = {}

    def build_from_directory(self, corpus_path: Path):
        for doc_id, (file_path, text) in enumerate(iter_documents(corpus_path)):
            tokens = list(tokenize(text))
            doc_len = len(tokens)
            self.documents[doc_id] = DocMeta(
                doc_id=doc_id,
                name=Path(file_path).name,
                length=doc_len
            )
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                self.postings[term].append(Posting(doc_id=doc_id, frequency=count))

        for term in self.postings:
            self.postings[term].sort(key=lambda p: p.doc_id)

    def save_pickle(self, filepath: Path):
        with open(filepath, "wb") as f:
            pickle.dump({"documents": self.documents, "postings": self.postings}, f)

    @classmethod
    def load_pickle(cls, filepath: Path) -> "InvertedIndex":
        # Коментар щодо безпеки: модуль pickle не захищений від шкідливого коду.
        # Завантаження чужих файлів (untrusted pickle) може призвести до виконання
        # довільних команд у системі (RCE) через метод __reduce__.
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx.documents = data["documents"]
        idx.postings = data["postings"]
        return idx

    def save_binary(self, filepath: Path):
        with open(filepath, "wb") as f:
            f.write(struct.pack("<I", len(self.documents)))
            for doc in self.documents.values():
                name_bytes = doc.name.encode("utf-8")
                f.write(struct.pack("<II", doc.doc_id, len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", doc.length))

            f.write(struct.pack("<I", len(self.postings)))
            for term, post_list in self.postings.items():
                term_bytes = term.encode("utf-8")
                f.write(struct.pack("<I", len(term_bytes)))
                f.write(term_bytes)
                n = len(post_list)
                f.write(struct.pack("<I", n))
                doc_ids = array.array("I", (p.doc_id for p in post_list))
                freqs = array.array("I", (p.frequency for p in post_list))
                doc_ids.tofile(f)
                freqs.tofile(f)

    @classmethod
    def load_binary(cls, filepath: Path) -> "InvertedIndex":
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
                doc_ids = array.array("I")
                doc_ids.fromfile(f, n)
                freqs = array.array("I")
                freqs.fromfile(f, n)
                idx.postings[term] = [
                    Posting(doc_id=doc_ids[i], frequency=freqs[i])
                    for i in range(n)
                ]
        return idx


def main():
    parser = argparse.ArgumentParser(description="Індексація текстового корпусу findex")
    parser.add_argument("corpus_dir", type=Path, help="Шлях до теки з корпусом текстів")
    parser.add_argument("--out", type=Path, default=Path("index.bin"), help="Вихідний файл індексу")
    parser.add_argument("--format", choices=["binary", "pickle"], default="binary", help="Формат збереження")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    index = InvertedIndex()
    index.build_from_directory(args.corpus_dir)

    if args.format == "binary":
        index.save_binary(args.out)
    else:
        index.save_pickle(args.out)

    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    file_size_kb = os.path.getsize(args.out) / 1024
    print(f"Індексація успішна!")
    print(f"Документів: {len(index.documents)}, унікальних термінів: {len(index.postings)}")
    print(f"Збережено у: {args.out} ({file_size_kb:.2f} КБ)")
    print(f"Час побудови та збереження: {elapsed_time:.4f} с")
    print(f"Пікова пам'ять: {peak_memory / (1024 * 1024):.2f} МБ")


if __name__ == "__main__":
    main()