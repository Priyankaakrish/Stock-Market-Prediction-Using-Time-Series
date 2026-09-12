.PHONY: help install data train train-fast test lint api mlflow docker clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install training dependencies
	pip install -r requirements-dev.txt

data:  ## Ingest -> clean -> features -> split
	python -m src.ingest && python -m src.preprocess && python -m src.features && python -m src.split

train:  ## Full training run (all 7 models)
	python -m src.train

train-fast:  ## CI mode: skip Prophet, shrink the LSTM
	python -m src.train --fast

test:  ## Run the test suite
	pytest tests -v

lint:  ## Ruff
	ruff check src api tests

api:  ## Serve the API locally on :8000
	uvicorn api.main:app --reload --port 8000

mlflow:  ## MLflow UI on :5002
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002

docker:  ## Build and start the whole stack
	docker compose -f deploy/docker-compose.yml up -d --build

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache data/processed/*.csv models/*.pkl reports/figures/*.png
