import torch
from .decoder import Decoder
from .data import load_text, build_vocab, encode, decode, make_splits, get_batch

torch.manual_seed(1337)

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

text = load_text('input.txt')
chars, stoi, itos, vocab_size = build_vocab(text)

# Train and test splits
data_tensor = torch.tensor(encode(text, stoi), dtype=torch.long)
train_data, val_data = make_splits(data_tensor, train_ratio=0.9)

# data loading wrapper closing over config and splits
def get_batch_local(split):
    return get_batch(split, train_data, val_data, block_size, batch_size, device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_local(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_loop():
    model = Decoder(vocab_size)
    model = model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch_local('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist(), itos))

__all__ = [
    'train_loop'
]
