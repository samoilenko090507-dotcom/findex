import time
import tracemalloc
from pathlib import Path

from findex.corpus import build_stats_eager, build_stats_stream


def run_benchmark():
    data_path = Path("data")
    if not data_path.exists() or not list(data_path.glob("*.txt")):
        print(
            "Папка data порожня або не знайдена. Спочатку запустіть download_data.py!"
        )
        return

    print("=== ЗАПУСК ПОРІВНЯННЯ (BENCHMARK) ===\n")

    # 1. Потокова версія через генератори
    tracemalloc.start()
    start_time = time.perf_counter()

    stream_docs, stream_tokens, stream_vocab, stream_top = build_stats_stream(data_path)

    stream_time = time.perf_counter() - start_time
    _, stream_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 2. Жадібна версія через повні списки в пам'яті
    tracemalloc.start()
    start_time = time.perf_counter()

    _eager_docs, _eager_tokens, _eager_vocab, _eager_top = build_stats_eager(data_path)

    eager_time = time.perf_counter() - start_time
    _, eager_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Переведення байтів у мегабайти
    stream_mb = stream_peak / (1024 * 1024)
    eager_mb = eager_peak / (1024 * 1024)

    print(f"Кількість знайдених документів: {stream_docs}")
    print(f"Загальна кількість токенів (слів): {stream_tokens}")
    print(f"Розмір унікального словника: {stream_vocab}\n")

    print("| Підхід | Час виконання (с) | Пікова пам'ять (МБ) |")
    print("| :--- | :--- | :--- |")
    print(f"| Потоковий (генератори / yield) | {stream_time:.4f} | {stream_mb:.2f} |")
    print(f"| Жадібний (списки в пам'яті) | {eager_time:.4f} | {eager_mb:.2f} |\n")

    print("Топ-10 найчастіших слів у корпусі:")
    for word, count in stream_top[:10]:
        print(f"  {word}: {count}")


if __name__ == "__main__":
    run_benchmark()
