PYTHON ?= python
PIP := $(PYTHON) -m pip

.PHONY: setup lint format test data train up down logs serve-local drift load-test dvc-repro

setup:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=80 -q

data:
	$(PYTHON) -m src.data.prepare

dvc-repro:
	dvc repro

train:
	$(PYTHON) -m src.models.train

serve-local:
	$(PYTHON) -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

drift:
	$(PYTHON) -m src.monitoring.report

load-test:
	$(PYTHON) scripts/load_test.py

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f
