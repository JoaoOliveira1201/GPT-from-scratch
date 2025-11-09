import torch
from . import config
from .tokenizer.bpe import BPE

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

tokenizer = BPE()
tokenizer.load("tokenizer/weights/bpe_weights.model")


def encode(s):
    return tokenizer.encode(s)


def decode(l):
    return tokenizer.decode(l)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(config.train_test_split * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data_tensor = train_data if split == "train" else val_data
    ix = torch.randint(len(data_tensor) - config.block_size, (config.batch_size,))
    x = torch.stack([data_tensor[i : i + config.block_size] for i in ix])
    y = torch.stack([data_tensor[i + 1 : i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y
