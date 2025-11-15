.PHONY: help train-tokenizer train inference print_model_info format run clean

help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  train-tokenizer - Train the tokenizer"
	@echo "  train - Train the model"
	@echo "  inference - Run inference"
	@echo "  print_model_info - Print model information"
	@echo "  format - Format the code"
	@echo "  run - Run everything"
	@echo "  clean - Clean the project"

train-tokenizer:
	uv run python -m src.tokenizer.train

train:
	uv run python -m main --train

inference:
	uv run python -m main --inference

print_model_info:
	uv run python -m main --print_model_info

format:
	uv run ruff check --select I --fix
	uv run ruff format .
	uv run ruff check .

run:
	uv run python -m main --train --inference --print_model_info

clean:
	find . -name "*.pyc" -type f -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf .ruff_cache/
	rm -rf logs/
	find . -type d -empty -delete