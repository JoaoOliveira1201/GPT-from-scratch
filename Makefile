.PHONY: train-tokenizer train inference print_model_info format run

train-tokenizer:
	uv run src/tokenizer/train.py

train:
	uv run main.py --train

inference:
	uv run main.py --inference

print_model_info:
	uv run main.py --print_model_info

format:
	uv run ruff format .

run:
	uv run main.py --train --inference --print_model_info

clean:
	find . -name "*.pyc" -type f -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	rm -rf .ruff_cache/*
	find . -type d -empty -delete