import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from findex.index import Index, open_index
from findex.scoring import BM25Scorer, Scorer, TfIdfScorer
from findex.search import ranked_search

app = typer.Typer(
    name="findex",
    help="findex: консольний інвертований індекс та ранжований пошук.",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


def setup_logging(verbose: int) -> None:
    if verbose == 0:
        level = logging.WARNING
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Рівень логування (-v для INFO, -vv для DEBUG).",
        ),
    ] = 0,
) -> None:
    setup_logging(verbose)


@app.command()
def index(
    corpus_dir: Annotated[
        Path, typer.Argument(help="Шлях до теки з документами корпусу.")
    ],
    output: Annotated[
        Path, typer.Option("-o", "--out", help="Файл збереження індексу.")
    ] = Path("index.bin"),
    positions: Annotated[
        bool,
        typer.Option("--positions/--no-positions", help="Зберігати позиції токенів."),
    ] = True,
) -> None:
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        err_console.print(f"Помилка: директорія корпусу '{corpus_dir}' не знайдена.")
        raise typer.Exit(code=1)

    files = list(corpus_dir.glob("*.txt"))
    if not files:
        err_console.print(f"Помилка: у теці '{corpus_dir}' немає файлів .txt.")
        raise typer.Exit(code=1)

    idx = Index()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Індексація файлів...", total=len(files))
        idx.build_from_directory(corpus_dir, store_positions=positions)
        progress.update(task, completed=len(files))

    if str(output).endswith(".pkl"):
        idx.save_pickle(output)
    else:
        idx.save_binary(output)

    console.print(
        f"[bold green]Успіх:[/bold green] збережено {len(idx)} слів із {idx.num_docs} документів у '{output}'."
    )


@app.command()
def search(
    index_path: Annotated[
        Path, typer.Argument(help="Шлях до файлу індексу (.bin або .pkl).")
    ],
    query: Annotated[str, typer.Argument(help="Пошуковий запит.")],
    scorer: Annotated[
        Literal["bm25", "tfidf"], typer.Option("--scorer", help="Модель ранжування.")
    ] = "bm25",
    top: Annotated[
        int, typer.Option("--top", "-k", help="Кількість топових результатів.")
    ] = 5,
    corpus_dir: Annotated[
        Path, typer.Option("--corpus", help="Шлях до теки документів для сніпетів.")
    ] = Path("data"),
    as_json: Annotated[
        bool, typer.Option("--json", help="Вивести результат у чистому форматі JSON.")
    ] = False,
) -> None:
    if not index_path.exists():
        err_console.print(f"Помилка: файл індексу '{index_path}' не знайдено.")
        raise typer.Exit(code=1)

    scorer_obj: Scorer = BM25Scorer() if scorer == "bm25" else TfIdfScorer()

    try:
        with open_index(index_path) as idx:
            results = ranked_search(idx, query, scorer_obj, corpus_dir, top_k=top)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"Помилка виконання пошуку: {exc}")
        raise typer.Exit(code=1)

    if as_json:
        data = [
            {
                "rank": rank,
                "score": score,
                "doc_id": doc_id,
                "name": name,
                "snippet": snippet,
            }
            for rank, (score, doc_id, name, snippet) in enumerate(results, start=1)
        ]
        sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        return

    if not results:
        console.print("[yellow]Нічого не знайдено за заданим запитом.[/yellow]")
        return

    table = Table(title=f"Результати пошуку ({scorer.upper()}, top-{top})")
    table.add_column("№", style="cyan", justify="right")
    table.add_column("Документ", style="green")
    table.add_column("Doc ID", justify="right")
    table.add_column("Бал", justify="right", style="magenta")
    table.add_column("Сніпет", style="dim")

    for rank, (score, doc_id, name, snippet) in enumerate(results, start=1):
        table.add_row(str(rank), name, str(doc_id), f"{score:.4f}", snippet)

    console.print(table)


@app.command()
def stats(
    index_path: Annotated[
        Path, typer.Argument(help="Шлях до файлу індексу (.bin або .pkl).")
    ],
) -> None:
    if not index_path.exists():
        err_console.print(f"Помилка: файл індексу '{index_path}' не знайдено.")
        raise typer.Exit(code=1)

    with open_index(index_path) as idx:
        table = Table(title="Статистика індексу")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значення", style="green")

        table.add_row("Унікальних слів (термінів)", str(len(idx)))
        table.add_row("Кількість документів", str(idx.num_docs))
        table.add_row("Середня довжина документа", f"{idx.avg_doc_length:.2f}")

        console.print(table)


if __name__ == "__main__":
    app()
