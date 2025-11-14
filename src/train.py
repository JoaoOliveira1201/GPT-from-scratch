import logging
import time

import torch

from .configs import model as model_config
from .configs import training as training_config
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
            logger.debug(
                f"  {split} batch {k + 1}/{training_config.eval_iters}: loss={loss.item():.4f}"
            )

        out[split] = losses.mean()
        logger.debug(f"  {split} average loss: {out[split]:.4f}")

    model.train()
    return out


def train_model(tokenizer_path, data_files, model_save_path=None):
    logger.info("Starting model training process")
    start_time = time.time()

    data_loader = DataLoader(tokenizer_path)
    data_loader.load_data(data_files)

    model = GPTLanguageModel(data_loader.vocab_size).to(model_config.device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model initialized: {total_params / 1e6:.3f}M parameters, device={model_config.device}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)

    best_val_loss = float("inf")
    training_start = time.time()

    for iter in range(training_config.max_iters):
        if (
            iter % training_config.eval_interval == 0
            or iter == training_config.max_iters - 1
        ):
            losses = estimate_loss(model, data_loader)

            train_loss = losses["train"]
            val_loss = losses["val"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(
                    f"step {iter}: train {train_loss:.4f}, val {val_loss:.4f} (NEW BEST)"
                )
            else:
                logger.info(f"step {iter}: train {train_loss:.4f}, val {val_loss:.4f}")

        xb, yb = data_loader.get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % (training_config.eval_interval // 4) == 0:
            elapsed = time.time() - training_start
            progress = (iter + 1) / training_config.max_iters
            eta = elapsed / progress - elapsed if progress > 0 else 0
            logger.info(
                f"step {iter}: training... ({progress:.1%} complete, ETA: {eta:.0f}s)"
            )
    total_training_time = time.time() - start_time
    logger.info(
        f"Training completed in {total_training_time:.2f}s ({total_training_time / 60:.1f}min)"
    )

    save_path = model_save_path or training_config.model_path
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

    return model, data_loader.decode
