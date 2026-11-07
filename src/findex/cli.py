import typer
import uvicorn
from findex.crawler import main_crawl
from findex.web.config import settings

app = typer.Typer(name="findex", help="findex – локальний пошуковий рушій")

@app.command()
def crawl(
    url: str = typer.Argument(..., help="Початковий URL для збору"),
    max_pages: int = typer.Option(200, "--max-pages", help="Максимальна кількість сторінок"),
    concurrency: int = typer.Option(5, "--concurrency", help="Кількість паралельних воркерів"),
    delay: float = typer.Option(0.5, "--delay", help="Мінімальна пауза між запитами на хост у секундах"),
    output: str = typer.Option("crawled_pages.jsonl", "--output", help="Шлях до файлу JSONL"),
):
    """Асинхронний збір сторінок за допомогою crawler."""
    main_crawl(
        start_url=url,
        max_pages=max_pages,
        concurrency=concurrency,
        per_host_delay=delay,
        output_path=output,
    )

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
