import pytest
from findex.corpus import tokenize


def test_tokenize_basic():
    text = "Hello, world!"
    tokens = list(tokenize(text))
    assert tokens == ["hello", "world"]


def test_tokenize_normalization():
    # Нормалізація регістру та збереження слів з апострофами й дефісами
    text = "State-of-the-art AI doesn't fail!"
    tokens = list(tokenize(text))
    assert tokens == ["state-of-the-art", "ai", "doesn't", "fail"]


def test_tokenize_numbers():
    text = "Room 404 in 2026."
    tokens = list(tokenize(text))
    assert tokens == ["room", "404", "in", "2026"]