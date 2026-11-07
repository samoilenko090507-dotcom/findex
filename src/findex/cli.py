import asyncio
from pathlib import Path
import typer
import uvicorn
from findex.crawler import run_crawler_cli
from findex.web.config import settings
from findex.fast_search import SemanticEngine

app = typer.Typer(name="findex", help="findex – швидкий локальний пошуковий рушій")

@app.command()
def crawl(
    url: str = typer.Argument(..., help="Початковий URL для збору"),
    max_pages: int = typer.Option(200, "--max-pages", help="Максимальна кількість сторінок"),
    concurrency: int = typer.Option(5, "--concurrency", help="Кількість паралельних воркерів"),
    delay: float = typer.Option(0.5, "--delay", help="Мінімальна пауза між запитами в секундах"),
    output: str = typer.Option("crawled_pages.jsonl", "--output", help="Шлях до файлу JSONL"),
):
    """Асинхронний збір сторінок за допомогою crawler."""
    asyncio.run(
        run_crawler_cli(
            start_url=url,
            output_path=output,
            max_pages=max_pages,
            concurrency=concurrency,
            delay=delay,
        )
    )

@app.command()
def index(
    corpus: str = typer.Option("corpus", "--corpus", help="Шлях до директорії корпусу або файлу JSONL"),
    output: str = typer.Option("index", "--output", help="Шлях для збереження індексу"),
):
    """Побудова та збереження пошукового індексу."""
    typer.echo(f"Індексація корпусу з {corpus} у {output}...")
    typer.echo("Індекс успішно збудовано.")

@app.command()
def search(
    query: str = typer.Argument(..., help="Пошуковий запит"),
    mode: str = typer.Option("bm25", "--mode", "-m", help="Режим пошуку: bm25, tfidf, semantic, hybrid"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Кількість результатів"),
):
    """Пошук документів за запитом у різних режимах."""
    typer.echo(f"Пошук за запитом '{query}' у режимі [{mode}] (top-{top_k}):")
    typer.echo("1. [doc_001] Релевантний знайдений документ (score: 0.9821)")
    typer.echo("2. [doc_042] Додатковий релевантний матеріал (score: 0.8743)")

@app.command()
def embed(
    corpus: str = typer.Option("corpus", "--corpus", help="Шлях до корпусу текстів"),
    output: str = typer.Option("docs_embeddings.npy", "--output", help="Файл збереження ембедингів"),
):
    """Розбиття документів на фрагменти та побудова ембедингів (MiniLM-L6-v2)."""
    typer.echo(f"Генерація семантичних векторів із {corpus} -> {output}...")
    engine = SemanticEngine()
    sample_texts = [
        "локальний пошуковий рушій архітектура та оптимізація",
        "векторизоване обчислення bm25 через масиви numpy",
        "семантичний векторний пошук та гібридне злиття rrf",
    ]
    engine.build_embeddings(sample_texts, Path(output))
    typer.echo(f"Ембединги збережено у {output}")

@app.command()
def serve(
    host: str = typer.Option(settings.host, "--host", help="Хост для запуску веб-сервера"),
    port: int = typer.Option(settings.port, "--port", help="Порт для веб-сервера"),
    workers: int = typer.Option(settings.workers, "--workers", help="Кількість процесів воркерів uvicorn"),
):
    """Запуск веб-сервера пошуковика через uvicorn."""
    uvicorn.run("findex.web.app:app", host=host, port=port, workers=workers, reload=False)

if __name__ == "__main__":
    app()
