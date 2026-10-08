import os
import re
import unicodedata
from collections import Counter
from pathlib import Path


def iter_documents(data_dir: str | Path):
    """Генератор, який читає текстові файли по одному, не завантажуючи весь корпус у пам'ять."""
    path = Path(data_dir)
    for entry in path.glob("*.txt"):
        with open(entry, "r", encoding="utf-8", errors="ignore") as f:
            yield entry.name, f.read()


def tokenize(text: str):
    """Потоковий токенізатор із нормалізацією Unicode.

    Правила:
    1. Приведення до NFKC (єдиний стандарт символів).
    2. Нижній регістр (case-folding).
    3. Слова з літерами, дефісами та апострофами.
    4. Числа залишаються токенами.
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    pattern = re.compile(r"\b[a-zа-яіїєґ0-9]+(?:['’\-][a-zа-яіїєґ0-9]+)*\b")
    for match in pattern.finditer(normalized):
        yield match.group(0)


def build_stats_stream(data_dir: str | Path):
    """Потокова обробка через генератори (економна до пам'яті)."""
    doc_count = 0
    token_count = 0
    vocab = Counter()

    for _, content in iter_documents(data_dir):
        doc_count += 1
        for token in tokenize(content):
            token_count += 1
            vocab[token] += 1

    return doc_count, token_count, len(vocab), vocab.most_common(50)


def build_stats_eager(data_dir: str | Path):
    """Жадібна обробка: спочатку все вантажиться у списки в пам'ять."""
    path = Path(data_dir)
    all_docs = []
    for entry in path.glob("*.txt"):
        with open(entry, "r", encoding="utf-8", errors="ignore") as f:
            all_docs.append((entry.name, f.read()))

    all_tokens = []
    for _, content in all_docs:
        all_tokens.extend(list(tokenize(content)))

    vocab = Counter(all_tokens)
    return len(all_docs), len(all_tokens), len(vocab), vocab.most_common(50)