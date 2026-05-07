FROM python:3.12-slim AS builder

WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --upgrade pip wheel && pip install --prefix=/install .

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src /app/src
COPY eval /app/eval
COPY alembic /app/alembic
COPY alembic.ini /app/alembic.ini
COPY workspace /app/workspace

EXPOSE 8000
CMD ["uvicorn", "agentic_runner.api:app", "--host", "0.0.0.0", "--port", "8000"]
