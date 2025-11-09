.PHONY: train-tokenizer

train-tokenizer:
	uv run src/tokenizer/bpe.py

format:
	uv run ruff format .