import logging
import torch
from .configs import training as training_config
from .configs import model as model_config
from .data import DataLoader
from .model import GPTLanguageModel

logger = logging.getLogger(__name__)


@torch.no_grad()
def estimate_loss(model, data_loader):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(training_config.eval_iters)
        for k in range(training_config.eval_iters):
            X, Y = data_loader.get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(tokenizer_path, data_files, model_save_path=None):
    data_loader = DataLoader(tokenizer_path)
    data_loader.load_data(data_files)

    model = GPTLanguageModel(data_loader.vocab_size).to(model_config.device)
    logger.info(
        f"parameters={sum(p.numel() for p in model.parameters()) / 1e6:.3f}M, device={model_config.device}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

    for iter in range(training_config.max_iters):
        if (
            iter % training_config.eval_interval == 0
            or iter == training_config.max_iters - 1
        ):
            losses = estimate_loss(model, data_loader)
            logger.info(
                f"step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}"
            )

        xb, yb = data_loader.get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    save_path = model_save_path or training_config.model_path
    logger.info(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
