import argparse
import logging

import torch

from src.configs import logging as logging_config
from src.configs import model as model_config
from src.configs import training as training_config
from src.data import DataLoader
from src.inference import run_inference
from src.model import GPTLanguageModel
from src.train import train_model

logger = logging.getLogger(__name__)


def train():
    try:
        model, decode_fn = train_model(
            tokenizer_path="weights/bpe_weights/bpe_weights.model",
            data_files=training_config.gpt_files,
        )
        return model, decode_fn
    except FileNotFoundError as e:
        logger.error(f"Training failed: Required file not found - {e}")
        raise
    except Exception as e:
        logger.error(f"Training failed with unexpected error: {e}")
        raise


def inference():
    try:
        data_loader = DataLoader("weights/bpe_weights/bpe_weights.model")
        model = GPTLanguageModel(data_loader.vocab_size)

        try:
            model.load_state_dict(torch.load(training_config.model_path))
        except FileNotFoundError:
            logger.error(f"Model file not found at {training_config.model_path}")
            raise

        model.to(model_config.device)
        model.eval()

        text = run_inference(model, data_loader.decode, max_new_tokens=500)
        print(text)

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def print_model_info():
    try:
        data_loader = DataLoader("weights/bpe_weights/bpe_weights.model")
        model = GPTLanguageModel(data_loader.vocab_size)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Model created with vocab_size={data_loader.vocab_size}, {total_params / 1e6:.3f}M parameters"
        )
        print(model)
        print(f"parameters={total_params / 1e6:.3f}M, device={model_config.device}")

    except Exception as e:
        logger.error(f"Failed to print model info: {e}")
        raise


if __name__ == "__main__":
    logging_config.setup_logging()

    parser = argparse.ArgumentParser(description="GPT from scratch")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--inference", action="store_true", help="Run inference")
    parser.add_argument(
        "--print_model_info", action="store_true", help="Print model information"
    )

    args = parser.parse_args()

    if args.train:
        train()
    if args.inference:
        inference()
    if args.print_model_info:
        print_model_info()
