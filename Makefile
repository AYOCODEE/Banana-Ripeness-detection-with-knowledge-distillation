.PHONY: install install-dev test lint format clean run-api docker-build docker-run

## Install production dependencies
install:
	pip install -r requirements.txt

## Install development dependencies
install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

## Run unit tests
test:
	pytest tests/ -v --tb=short

## Run tests with coverage report
test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

## Lint the code with flake8
lint:
	flake8 src/ tests/ app.py --max-line-length=100 --extend-ignore=E203,W503

## Format the code with black
format:
	black src/ tests/ app.py

## Check formatting without modifying files
format-check:
	black --check src/ tests/ app.py

## Remove Python cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

## Run the FastAPI inference server
run-api:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

## Run inference on a single image (usage: make infer MODEL=path IMAGE=path)
infer:
	python -m src.inference --model $(MODEL) --image $(IMAGE)

## Build the Docker image
docker-build:
	docker build -t banana-ripeness .

## Run the Docker container
docker-run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models banana-ripeness
