# Makefile for DroneVision development and deployment

.PHONY: help install lint format test docker-build docker-run clean

help:
	@echo "Available commands:"
	@echo "  install      Install package in editable mode with development dependencies"
	@echo "  lint         Run ruff checks and type verification"
	@echo "  format       Auto-format code using ruff"
	@echo "  test         Run unit tests"
	@echo "  docker-build Build the Gradio demo application Docker image"
	@echo "  docker-run   Run the Gradio demo application Docker container"
	@echo "  clean        Remove temporary files, logs, and caches"

install:
	pip install -e .[dev,api]

lint:
	ruff check .
	mypy --ignore-missing-imports dronevision/

format:
	ruff format .

test:
	pytest tests/ -v

docker-build:
	docker build -t dronevision:latest .

docker-run:
	docker run -p 7860:7860 dronevision:latest

clean:
	rm -rf `find . -name __pycache__`
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
