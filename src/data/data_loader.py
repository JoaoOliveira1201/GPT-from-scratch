import logging
import os
import torch
from typing import List

from ..configs import training as training_config
from ..configs import model as model_config
from ..tokenizer.bpe import BPE

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, tokenizer_path=None):
        self.tokenizer = None
        self.train_data = None
        self.val_data = None
        self.vocab_size = None

        if tokenizer_path:
            self.load_tokenizer(tokenizer_path)

    def load_tokenizer(self, tokenizer_path):
        self.tokenizer = BPE()
        self.tokenizer.load(tokenizer_path)
        self.vocab_size = 256 + len(self.tokenizer.merges)

    def load_data(self, file_paths: List[str]):

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded before loading data")

        text_chunks = []

        for path in file_paths:
            if os.path.isfile(path):
                text_chunks.append(self._read_file(path))
            elif os.path.isdir(path):
                text_chunks.append(self._read_directory(path))
            else:
                logger.warning(f"Path not found or invalid type: {path}")

        # Filter out None or empty strings
        valid_chunks = [t for t in text_chunks if t]

        if not valid_chunks:
            raise ValueError("No text content found in input files")

        # Join all chunks efficiently
        combined_text = "".join(valid_chunks)

        logger.info(f"Encoding {len(combined_text)} characters of text...")
        data = torch.tensor(self.encode(combined_text), dtype=torch.long)

        # Split data
        n = int(training_config.train_test_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

        logger.info(
            f"Data loaded. Train: {len(self.train_data)}, Val: {len(self.val_data)} tokens"
        )

    def _read_file(self, file_path: str) -> str:

        allowed_exts = ('.csv', '.txt')
        if not file_path.endswith(allowed_exts):
            logger.debug(f"Skipping unsupported file type: {file_path}")
            return ""

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file '{file_path}': {e}")
            return ""

    def _read_directory(self, dir_path: str) -> str:

        dir_texts = []
        logger.info(f"Scanning directory: {dir_path}")

        for root, _, files in os.walk(dir_path):
            for file in files:
                full_path = os.path.join(root, file)
                content = self._read_file(full_path)
                if content:
                    dir_texts.append(content)

        return "".join(dir_texts)

    def encode(self, text):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        return self.tokenizer.encode(text)

    def decode(self, token_ids):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        return self.tokenizer.decode(token_ids)

    def get_batch(self, split):
        if self.train_data is None or self.val_data is None:
            raise ValueError("Data not loaded")

        data_tensor = self.train_data if split == "train" else self.val_data

        # Ensure data is large enough for block_size
        if len(data_tensor) <= model_config.block_size:
            raise ValueError(f"Dataset too small for block_size {model_config.block_size}")

        max_start_idx = len(data_tensor) - model_config.block_size
        ix = torch.randint(max_start_idx, (training_config.batch_size,))

        x = torch.stack([data_tensor[i: i + model_config.block_size] for i in ix])
        y = torch.stack(
            [data_tensor[i + 1: i + model_config.block_size + 1] for i in ix]
        )

        x, y = x.to(model_config.device), y.to(model_config.device)

        return x, y