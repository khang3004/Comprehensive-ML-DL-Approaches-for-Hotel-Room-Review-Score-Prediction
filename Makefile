.PHONY: help install dev lint format test clean docker-build docker-run

help:
@echo "Available commands:"
@echo "  make install        - Install dependencies"
@echo "  make dev            - Install development dependencies"
@echo "  make lint           - Run linting checks"
@echo "  make format         - Format code with black and isort"
@echo "  make test           - Run tests"
@echo "  make clean          - Clean build artifacts"
@echo "  make docker-build   - Build Docker image"
@echo "  make docker-run     - Run Docker container"

install:
pip install -r requirements.txt
pip install -e .

dev: install
pip install -r requirements-dev.txt
pre-commit install

lint:
flake8 src/ tests/ --max-line-length=120 --ignore=E203,W503
black src/ tests/ --check
isort src/ tests/ --check-only
mypy src/ --ignore-missing-imports

format:
black src/ tests/
isort src/ tests/

test:
pytest tests/ -v --cov=src --cov-report=html

clean:
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type d -name "*.egg-info" -exec rm -rf {} +
rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
rm -rf logs/* results/*

docker-build:
docker build -t booking-hotel-analytics:latest .

docker-run:
docker run -it --rm \
-v $(pwd)/data:/app/data \
-v $(pwd)/models:/app/models \
-v $(pwd)/results:/app/results \
booking-hotel-analytics:latest bash

docker-gpu:
docker run --gpus all -it --rm \
-v $(pwd)/data:/app/data \
-v $(pwd)/models:/app/models \
-v $(pwd)/results:/app/results \
booking-hotel-analytics:latest

all: lint test
