from typing import Literal
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    q: str = Field(..., min_length=1, description="Пошуковий запит")
    k: int = Field(10, ge=1, le=100, description="Кількість результатів на сторінку")
    scorer: Literal["bm25", "tfidf"] = Field("bm25", description="Функція ранжування")
    page: int = Field(1, ge=1, description="Номер сторінки")

class SearchHit(BaseModel):
    doc_id: int
    score: float
    title: str
    snippet: str
    url: str | None = None

class SearchResponse(BaseModel):
    query: str
    total_hits: int
    page: int
    total_pages: int
    results: list[SearchHit]

class DocumentResponse(BaseModel):
    doc_id: int
    title: str
    text: str
    url: str | None = None

class StatsResponse(BaseModel):
    documents_count: int
    terms_count: int
    index_size_bytes: int

class HealthResponse(BaseModel):
    status: Literal["ok", "starting"]
    index_loaded: bool
