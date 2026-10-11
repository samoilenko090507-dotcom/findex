import os
import platform
import statistics
import time
from pathlib import Path
import matplotlib.pyplot as plt
import psutil

from findex.index import build_index


def run_single_benchmark(
    corpus_dir: Path, workers: int, executor: str
) -> tuple[float, float, float]:
    proc = psutil.Process()
    # Початкові заміри
    cpu_start = proc.cpu_times().user + proc.cpu_times().system
    wall_start = time.perf_counter()

    # Запуск індексації
    build_index(corpus_dir, workers=workers, executor=executor)

    wall_time = time.perf_counter() - wall_start
    cpu_time = (proc.cpu_times().user + proc.cpu_times().system) - cpu_start
    rss_mb = proc.memory_info().rss / (1024 * 1024)

    return wall_time, cpu_time, rss_mb


def benchmark_suite(corpus_dir: Path):
    cpu_count = os.cpu_count() or 4
    workers_list = sorted(list({1, 2, 4, 8, cpu_count}))

    print(f"Процесор: {platform.processor()} | Логічних ядер: {cpu_count}")
    print(f"Корпус: {corpus_dir.resolve()}")
    print("Виконуємо заміри (по 3 прогони на кожну конфігурацію)...")

    results = []

    # 1. Базовий послідовний варіант (1 воркер)
    times_serial, cpus_serial, rss_serial = [], [], []
    for _ in range(3):
        w, c, r = run_single_benchmark(corpus_dir, workers=1, executor="serial")
        times_serial.append(w)
        cpus_serial.append(c)
        rss_serial.append(r)

    base_time = statistics.median(times_serial)
    results.append(
        {
            "mode": "serial",
            "workers": 1,
            "wall_time": base_time,
            "cpu_time": statistics.median(cpus_serial),
            "rss_mb": statistics.median(rss_serial),
            "speedup": 1.0,
        }
    )

    # 2. Процеси та Потоки
    for executor in ["processes", "threads"]:
        for w in workers_list:
            if w == 1 and executor == "threads":
                continue
            w_times, c_times, r_times = [], [], []
            for _ in range(3):
                wall_t, cpu_t, rss_val = run_single_benchmark(
                    corpus_dir, workers=w, executor=executor
                )
                w_times.append(wall_t)
                c_times.append(cpu_t)
                r_times.append(rss_val)

            med_wall = statistics.median(w_times)
            speedup = base_time / med_wall if med_wall > 0 else 0.0
            results.append(
                {
                    "mode": executor,
                    "workers": w,
                    "wall_time": med_wall,
                    "cpu_time": statistics.median(c_times),
                    "rss_mb": statistics.median(r_times),
                    "speedup": speedup,
                }
            )

    # Вивід підсумкової таблиці для README
    print("\n| Режим | Воркери | Wall time (s) | CPU time (s) | RSS (MB) | Прискорення |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['mode']} | {r['workers']} | {r['wall_time']:.3f} | "
            f"{r['cpu_time']:.3f} | {r['rss_mb']:.1f} | {r['speedup']:.2f}x |"
        )

    # Побудова графіка прискорення
    proc_data = [r for r in results if r["mode"] == "processes"]
    thread_data = [r for r in results if r["mode"] == "threads"]

    plt.figure(figsize=(9, 6))

    # Ідеальна пряма
    max_w = max(workers_list)
    plt.plot([1, max_w], [1, max_w], "k--", label="Ідеальне лінійне прискорення")

    if proc_data:
        x_proc = [r["workers"] for r in proc_data]
        y_proc = [r["speedup"] for r in proc_data]
        plt.plot(x_proc, y_proc, "o-", color="#1f77b4", label="multiprocessing")

    if thread_data:
        x_th = [r["workers"] for r in thread_data]
        y_th = [r["speedup"] for r in thread_data]
        plt.plot(x_th, y_th, "s-", color="#ff7f0e", label="threading (під GIL)")

    plt.title("Прискорення індексації залежно від кількості воркерів")
    plt.xlabel("Кількість воркерів")
    plt.ylabel("Прискорення (відносно 1 ядра)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_plot = Path("speedup.png")
    plt.savefig(out_plot, dpi=300)
    print(f"\nГрафік успішно збережено у {out_plot.resolve()}")


if __name__ == "__main__":
    data_path = Path("data")
    if not data_path.exists():
        print("Тека data не знайдена!")
    else:
        benchmark_suite(data_path)