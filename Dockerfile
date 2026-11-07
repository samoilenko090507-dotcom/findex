# Етап 1: Збірка середовища
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Налаштовуємо uv для компіляції та ізоляції
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Копіюємо конфігурації залежностей
COPY pyproject.toml uv.lock ./

# Встановлюємо залежності без самого проєкту для кешування шару
RUN uv sync --frozen --no-install-project --no-dev

# Копіюємо вихідний код
COPY . .

# Фінальна синхронізація проєкту
RUN uv sync --frozen --no-dev

# Етап 2: Фінальний робочий образ
FROM python:3.12-slim-bookworm

WORKDIR /app

# Копіюємо віртуальне оточення та файли проєкту
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Додаємо віртуальне середовище в PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PORT=8000

EXPOSE 8000

# Запуск сервера
CMD ["uvicorn", "findex.web.app:app", "--host", "0.0.0.0", "--port", "8000"]