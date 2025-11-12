import torch
from .configs import training as training_config
from .configs import model as model_config
from .tokenizer.bpe import BPE


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

    def load_data(self, file_paths):
        combined_text = ""
        for file_path in file_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    combined_text += f.read()
            except FileNotFoundError:
                raise FileNotFoundError(f"File '{file_path}' not found")
            except Exception as e:
                raise Exception(f"Error reading file '{file_path}': {e}")

        if not combined_text:
            raise ValueError("No text content found in input files")

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be loaded before loading data")

        data = torch.tensor(self.encode(combined_text), dtype=torch.long)
        n = int(training_config.train_test_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

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
        ix = torch.randint(
            len(data_tensor) - model_config.block_size, (training_config.batch_size,)
        )
        x = torch.stack([data_tensor[i : i + model_config.block_size] for i in ix])
        y = torch.stack(
            [data_tensor[i + 1 : i + model_config.block_size + 1] for i in ix]
        )
        x, y = x.to(model_config.device), y.to(model_config.device)
        return x, y
