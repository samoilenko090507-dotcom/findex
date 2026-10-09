import logging
import re
from collections.abc import Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    for match in WORD_RE.finditer(text):
        yield match.group(0).lower()


def iter_documents(corpus_path: Path) -> Iterator[tuple[str, str]]:
    if not corpus_path.exists():
        logger.warning("Шлях до корпусу не існує: %s", corpus_path)
        return

    for file_path in corpus_path.glob("*.txt"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            yield str(file_path), text
        except OSError as exc:
            logger.error("Не вдалося прочитати файл %s: %s", file_path, exc)
