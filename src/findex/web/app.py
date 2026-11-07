import logging
import math
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from findex.index import Index, iter_documents
from findex.search import BM25Scorer, TfIdfScorer, ranked_search
from findex.web.config import settings
from findex.web.schemas import (
    DocumentResponse,
    HealthResponse,
    SearchHit,
    SearchQuery,
    SearchResponse,
    StatsResponse,
)

logger = logging.getLogger("findex.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

class AppState:
    index: Index | None = None

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.index_path.exists():
        logger.info(f"Завантаження індексу з {settings.index_path}...")
        try:
            if str(settings.index_path).endswith(".pkl"):
                state.index = Index.load_pickle(settings.index_path)
            else:
                state.index = Index.load_binary(settings.index_path)
            logger.info("Індекс успішно завантажено в пам'ять.")
        except Exception as e:
            logger.error(f"Не вдалося завантажити індекс: {e}")
            state.index = None
    else:
        logger.warning(f"Файл індексу {settings.index_path} не знайдено під час запуску.")
        state.index = None
    yield
    state.index = None

app = FastAPI(title="findex Web Search", version="0.7.0", lifespan=lifespan)

@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Unhandled server error [request_id={request_id}]: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal Server Error", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )

def get_index() -> Index:
    if state.index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Index not loaded",
        )
    return state.index

@app.get("/health", response_model=HealthResponse)
def health(index: Annotated[Index, Depends(get_index)]):
    return HealthResponse(status="ok", index_loaded=True)

@app.get("/stats", response_model=StatsResponse)
def stats(index: Annotated[Index, Depends(get_index)]):
    doc_count = getattr(index, "num_docs", 0)
    term_count = len(index) if hasattr(index, "__len__") else len(getattr(index, "_postings", {}))
    size_bytes = settings.index_path.stat().st_size if settings.index_path.exists() else 0
    return StatsResponse(
        documents_count=doc_count,
        terms_count=term_count,
        index_size_bytes=size_bytes,
    )

@app.get("/docs/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: int, index: Annotated[Index, Depends(get_index)]):
    docs_map = getattr(index, "documents", {})
    if doc_id not in docs_map:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Документ з doc_id={doc_id} не знайдено",
        )
    meta = docs_map[doc_id]
    title = getattr(meta, "name", f"Документ #{doc_id}")
    text = ""
    if settings.corpus_dir.exists():
        for cur_id, (_, content) in enumerate(iter_documents(settings.corpus_dir)):
            if cur_id == doc_id:
                text = content
                break
    return DocumentResponse(
        doc_id=doc_id,
        title=title,
        text=text,
        url=None,
    )

def execute_search(query: str, index: Index, scorer_name: str, k: int, page: int):
    scorer = BM25Scorer() if scorer_name.lower() == "bm25" else TfIdfScorer()
    raw_results = ranked_search(
        index=index,
        query_str=query,
        scorer=scorer,
        corpus_dir=settings.corpus_dir,
        top_k=page * k,
    )
    total_hits = len(raw_results)
    total_pages = max(1, math.ceil(total_hits / k))
    start_idx = (page - 1) * k
    end_idx = start_idx + k
    page_items = raw_results[start_idx:end_idx]
    
    hits = [
        SearchHit(
            doc_id=doc_id,
            score=round(float(score), 4),
            title=name,
            snippet=snippet,
            url=None,
        )
        for score, doc_id, name, snippet in page_items
    ]
    return total_hits, total_pages, hits

@app.get("/search", response_model=SearchResponse)
def search_endpoint(
    query_params: Annotated[SearchQuery, Query()],
    index: Annotated[Index, Depends(get_index)],
):
    total_hits, total_pages, results = execute_search(
        query=query_params.q,
        index=index,
        scorer_name=query_params.scorer,
        k=query_params.k,
        page=query_params.page,
    )
    return SearchResponse(
        query=query_params.q,
        total_hits=total_hits,
        page=query_params.page,
        total_pages=total_pages,
        results=results,
    )

@app.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    q: str = "",
    page: int = 1,
    scorer: str = "bm25",
    k: int = 10,
):
    results = None
    total_hits = 0
    total_pages = 1
    error = None

    if q.strip():
        if state.index is None:
            error = "Індекс не завантажено на сервері."
        else:
            try:
                total_hits, total_pages, results = execute_search(
                    query=q,
                    index=state.index,
                    scorer_name=scorer,
                    k=k,
                    page=page,
                )
            except Exception as e:
                error = f"Помилка пошуку: {e}"

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "q": q,
            "page": page,
            "scorer": scorer,
            "results": results,
            "total_hits": total_hits,
            "total_pages": total_pages,
            "error": error,
        },
    )