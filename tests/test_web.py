from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from findex.web.app import app, get_index, state

client = TestClient(app, raise_server_exceptions=False)

def test_health_503_when_no_index():
    state.index = None
    app.dependency_overrides.clear()
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "Index not loaded"

def test_health_200_when_index_loaded():
    mock_index = MagicMock()
    app.dependency_overrides[get_index] = lambda: mock_index
    try:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["index_loaded"] is True
    finally:
        app.dependency_overrides.clear()

def test_stats_endpoint():
    mock_index = MagicMock()
    mock_index.num_docs = 42
    mock_index.__len__.return_value = 15
    app.dependency_overrides[get_index] = lambda: mock_index
    try:
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["documents_count"] == 42
        assert data["terms_count"] == 15
    finally:
        app.dependency_overrides.clear()

def test_get_document_success_and_404():
    mock_index = MagicMock()
    mock_meta = MagicMock()
    mock_meta.name = "Тестовий документ"
    mock_index.documents = {1: mock_meta}
    app.dependency_overrides[get_index] = lambda: mock_index
    
    try:
        res_ok = client.get("/docs/1")
        assert res_ok.status_code == 200
        assert res_ok.json()["title"] == "Тестовий документ"

        res_404 = client.get("/docs/999")
        assert res_404.status_code == 404
    finally:
        app.dependency_overrides.clear()

def test_search_validation_422():
    mock_index = MagicMock()
    app.dependency_overrides[get_index] = lambda: mock_index
    try:
        res = client.get("/search?k=-5")
        assert res.status_code == 422
    finally:
        app.dependency_overrides.clear()

def test_search_success(monkeypatch):
    mock_index = MagicMock()
    import findex.web.app as web_app
    monkeypatch.setattr(
        web_app,
        "execute_search",
        lambda query, index, scorer_name, k, page: (1, 1, [
            web_app.SearchHit(
                doc_id=10,
                score=3.1415,
                title="Стаття про FastAPI",
                snippet="Фреймворк <mark>FastAPI</mark> швидкий",
                url=None,
            )
        ])
    )
    app.dependency_overrides[get_index] = lambda: mock_index

    try:
        res = client.get("/search?q=FastAPI&scorer=bm25")
        assert res.status_code == 200
        data = res.json()
        assert data["query"] == "FastAPI"
        assert data["total_hits"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["doc_id"] == 10
    finally:
        app.dependency_overrides.clear()

def test_home_page_html():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "findex" in res.text
