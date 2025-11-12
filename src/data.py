import logging
import torch
import psutil
import os
from .configs import training as training_config
from .configs import model as model_config
from .tokenizer.bpe import BPE

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
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = BPE()
        self.tokenizer.load(tokenizer_path)
        self.vocab_size = 256 + len(self.tokenizer.merges)
        logger.info(f"Tokenizer loaded with vocab_size={self.vocab_size}")

    def load_data(self, file_paths):
        logger.info(f"Loading data from files: {file_paths}")

        combined_text = ""
        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    combined_text += text
                    logger.info(f"Loaded {len(text)} characters from {file_path}")
            except FileNotFoundError:
                raise FileNotFoundError(f"File '{file_path}' not found")
            except Exception as e:
                raise Exception(f"Error reading file '{file_path}': {e}")

        if not combined_text:
            raise ValueError("No text content found in input files")

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded before loading data")

        logger.info(f"Encoding {len(combined_text)} characters of text")
        data = torch.tensor(self.encode(combined_text), dtype=torch.long)
        n = int(training_config.train_test_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        logger.info(
            f"Data split: train={len(self.train_data)}, val={len(self.val_data)} tokens"
        )

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

        logger.debug(f"Getting batch for {split} split")
        data_tensor = self.train_data if split == "train" else self.val_data

        # Sample random starting indices for batches
        max_start_idx = len(data_tensor) - model_config.block_size
        ix = torch.randint(max_start_idx, (training_config.batch_size,))
        logger.debug(f"Sampled {training_config.batch_size} batch indices from 0 to {max_start_idx}")

        # Create input sequences
        x = torch.stack([data_tensor[i : i + model_config.block_size] for i in ix])
        # Create target sequences (shifted by 1)
        y = torch.stack(
            [data_tensor[i + 1 : i + model_config.block_size + 1] for i in ix]
        )

        # Move to device
        x, y = x.to(model_config.device), y.to(model_config.device)

        # Log memory usage if available
        if torch.cuda.is_available() and model_config.device == 'cuda':
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
            logger.debug(f"GPU memory: allocated={memory_allocated:.1f}MB, reserved={memory_reserved:.1f}MB")

        logger.debug(f"Batch created: x.shape={x.shape}, y.shape={y.shape}")
        return x, y
