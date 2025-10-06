import torch
from . import config
from . import data as data_mod
from .model import GPTLanguageModel


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = data_mod.get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_loop():
    logger, log_file = config.init_logger()
    model = GPTLanguageModel().to(config.device)
    logger.info(f"parameters={sum(p.numel() for p in model.parameters())/1e6:.3f}M, device={config.device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model)
            logger.info(f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

        xb, yb = data_mod.get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    logger.info(f"Saving model to {config.model_path}")
    torch.save(model.state_dict(), config.model_path)
    return model, data_mod.decode


