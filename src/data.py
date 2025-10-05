import torch

def load_text(path: str = 'input.txt') -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    vocab_size = len(chars)
    return chars, stoi, itos, vocab_size

def encode(s: str, stoi: dict) -> list:
    return [stoi[c] for c in s]

def decode(indices: list, itos: dict) -> str:
    return ''.join([itos[i] for i in indices])

def make_splits(data_tensor: torch.Tensor, train_ratio: float = 0.9):
    n = int(train_ratio * len(data_tensor))
    train_data = data_tensor[:n]
    val_data = data_tensor[n:]
    return train_data, val_data

def get_batch(split: str,
              train_data: torch.Tensor,
              val_data: torch.Tensor,
              block_size: int,
              batch_size: int,
              device: str):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

__all__ = [
    'load_text', 'build_vocab', 'encode', 'decode', 'make_splits', 'get_batch'
]


