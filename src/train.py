import logging
import time
import torch
from .configs import training as training_config
from .configs import model as model_config
from .data import DataLoader
from .model import GPTLanguageModel

logger = logging.getLogger(__name__)


@torch.no_grad()
def estimate_loss(model, data_loader):
    logger.debug("Starting loss estimation")
    out = {}
    model.eval()
    start_time = time.time()

    for split in ["train", "val"]:
        logger.debug(f"Estimating loss for {split} split")
        losses = torch.zeros(training_config.eval_iters)
        for k in range(training_config.eval_iters):
            X, Y = data_loader.get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
            logger.debug(f"  {split} batch {k+1}/{training_config.eval_iters}: loss={loss.item():.4f}")

        out[split] = losses.mean()
        logger.debug(f"  {split} average loss: {out[split]:.4f}")

    model.train()
    elapsed = time.time() - start_time
    logger.debug(f"Loss estimation completed in {elapsed:.2f}s")
    return out


def train_model(tokenizer_path, data_files, model_save_path=None):
    logger.info("Starting model training process")
    start_time = time.time()

    data_loader = DataLoader(tokenizer_path)
    data_loader.load_data(data_files)
    logger.info(f"Data loaded: {len(data_loader.train_data)} train tokens, {len(data_loader.val_data)} val tokens")

    model = GPTLanguageModel(data_loader.vocab_size).to(model_config.device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model initialized: {total_params / 1e6:.3f}M parameters, device={model_config.device}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    logger.info(f"Optimizer initialized: AdamW with lr={training_config.learning_rate}")

    best_val_loss = float('inf')
    training_start = time.time()

    logger.info(f"Starting training for {training_config.max_iters} iterations")
    for iter in range(training_config.max_iters):
        iter_start_time = time.time()

        # Evaluate and log progress
        if (
            iter % training_config.eval_interval == 0
            or iter == training_config.max_iters - 1
        ):
            eval_start = time.time()
            losses = estimate_loss(model, data_loader)
            eval_time = time.time() - eval_start

            train_loss = losses['train']
            val_loss = losses['val']

            # Track best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"step {iter}: train {train_loss:.4f}, val {val_loss:.4f} (NEW BEST)")
            else:
                logger.info(f"step {iter}: train {train_loss:.4f}, val {val_loss:.4f}")
            logger.info(f"  Evaluation time: {eval_time:.2f}s")

        # Training step
        step_start = time.time()
        xb, yb = data_loader.get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step_time = time.time() - step_start

        # Log training progress periodically
        if iter % (training_config.eval_interval // 4) == 0:
            elapsed = time.time() - training_start
            progress = (iter + 1) / training_config.max_iters
            eta = elapsed / progress - elapsed if progress > 0 else 0
            logger.info(f"step {iter}: training... ({progress:.1%} complete, ETA: {eta:.0f}s)")
    total_training_time = time.time() - start_time
    logger.info(f"Training completed in {total_training_time:.2f}s ({total_training_time/60:.1f}min)")

    save_path = model_save_path or training_config.model_path
    logger.info(f"Saving model to {save_path}")
    save_start = time.time()
    torch.save(model.state_dict(), save_path)
    save_time = time.time() - save_start
    logger.info(f"Model saved to {save_path} in {save_time:.2f}s")

    return model, data_loader.decode
