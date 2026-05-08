.PHONY: dev migrate seed test lint typecheck eval eval-smoke bench-regress up serve cli clean format down test-all

PY := python
PIP := pip

dev:
	$(PIP) install -e ".[dev]"

migrate:
	alembic upgrade head

seed:
	$(PY) -m agentic_runner.cli seed

test:
	pytest -q tests/unit

test-all:
	RUN_INTEGRATION=1 pytest -q tests/

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

typecheck:
	mypy src/agentic_runner

eval:
	$(PY) -m agentic_runner.cli eval run --suite runner_v1 --provider fake --output eval/baselines/runner_v1_fake.json --suite-dir eval/suites

eval-smoke:
	$(PY) -m agentic_runner.cli eval smoke --suite runner_v1 --baseline eval/baselines/runner_v1_fake.json --suite-dir eval/suites

bench-regress:
	$(PY) -m agentic_runner.cli eval bench-regress --suite runner_v1 --baseline eval/baselines/runner_v1_fake.json --suite-dir eval/suites --max-drift 0.30

up:
	docker compose up -d

down:
	docker compose down -v

serve:
	uvicorn agentic_runner.api:app --reload --host 0.0.0.0 --port 8000

cli:
	$(PY) -m agentic_runner.cli --help

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
