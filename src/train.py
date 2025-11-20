import logging
import time

import torch
from torch.amp import autocast, GradScaler

from .configs import model as model_config
from .configs import training as training_config
from .data import DataLoader
from .logger import logger as mlflow_logger
from .model import GPTLanguageModel

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')

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
    model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model initialized: {total_params / 1e6:.3f}M parameters, device={model_config.device}"
    )

    # Log hyperparameters to MLflow
    mlflow_logger.log_params(
        {
            "vocab_size": data_loader.vocab_size,
            "n_embd": model_config.n_embd,
            "n_head": model_config.n_head,
            "n_layer": model_config.n_layer,
            "dropout": model_config.dropout,
            "bias": model_config.bias,
            "block_size": model_config.block_size,
            "batch_size": training_config.batch_size,
            "max_iters": training_config.max_iters,
            "eval_interval": training_config.eval_interval,
            "learning_rate": training_config.learning_rate,
            "eval_iters": training_config.eval_iters,
            "train_test_split": training_config.train_test_split,
            "device": model_config.device,
            "total_params": f"{total_params / 1e6:.3f}M",
            "tokenizer_path": tokenizer_path,
            "data_files": str(data_files),
        }
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, fused=True if model_config.device == "cuda" else False)
    scaler = GradScaler()

    best_val_loss = float("inf")
    training_start = time.time()

    for iter in range(training_config.max_iters):

        should_eval = (iter % training_config.eval_interval == 0) or (iter == training_config.max_iters - 1)

        if should_eval:
            losses = estimate_loss(model, data_loader)

            train_loss = losses["train"]
            val_loss = losses["val"]

            # Log metrics to MLflow
            metrics = {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            mlflow_logger.log_metrics(metrics, step=iter)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(
                    f"step {iter}: train {train_loss:.4f}, val {val_loss:.4f} (NEW BEST)"
                )
            else:
                logger.info(f"step {iter}: train {train_loss:.4f}, val {val_loss:.4f}")

        xb, yb = data_loader.get_batch("train")

        with autocast('cuda'):
            _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

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

    # Log final training metrics
    mlflow_logger.log_metrics(
        {
            "total_training_time_seconds": total_training_time,
            "total_training_time_minutes": total_training_time / 60,
            "best_val_loss": float(best_val_loss),
        }
    )

    save_path = model_save_path or training_config.model_path
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved locally to {save_path}")

    # Log model artifact to MLflow (using state dict file instead of model object)
    try:
        mlflow_logger.log_artifact(save_path, artifact_path="model")
        logger.info("Model artifact logged to MLflow")
    except Exception as e:
        logger.warning(f"Could not log model artifact to MLflow: {e}")

    return model, data_loader.decode
