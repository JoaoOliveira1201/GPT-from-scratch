import argparse
import logging
import torch
from src.train import train_model
from src.inference import run_inference
from src.model import GPTLanguageModel
from src.data import DataLoader
from src.configs import training as training_config
from src.configs import model as model_config

logger = logging.getLogger(__name__)


def train():
    logger.info("Starting model training")
    model, decode_fn = train_model(
        tokenizer_path="weights/bpe_weights/bpe_weights.model",
        data_files=training_config.gpt_files,
    )
    logger.info("Training completed successfully")
    return model, decode_fn


def inference():
    logger.info("Starting inference")

    data_loader = DataLoader("weights/bpe_weights/bpe_weights.model")
    model = GPTLanguageModel(data_loader.vocab_size)
    logger.info(f"Loading model from {training_config.model_path}")
    model.load_state_dict(torch.load(training_config.model_path))
    model.to(model_config.device)
    model.eval()
    logger.info(f"Model loaded and moved to {model_config.device}")

    text = run_inference(model, data_loader.decode, max_new_tokens=500)
    logger.info("Inference completed successfully")
    print(text)


def print_model_info():
    logger.info("Printing model information")

    data_loader = DataLoader("weights/bpe_weights/bpe_weights.model")
    model = GPTLanguageModel(data_loader.vocab_size)
    logger.info(f"Model created with vocab_size={data_loader.vocab_size}")
    print(model)
    print(
        f"parameters={sum(p.numel() for p in model.parameters()) / 1e6:.3f}M, device={model_config.device}"
    )


if __name__ == "__main__":
    logger.info("GPT from scratch application started")

    parser = argparse.ArgumentParser(description="GPT from scratch")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--inference", action="store_true", help="Run inference")
    parser.add_argument(
        "--print_model_info", action="store_true", help="Print model information"
    )

    args = parser.parse_args()

    if args.train:
        logger.info("Train mode selected")
        train()
    elif args.inference:
        logger.info("Inference mode selected")
        inference()
    elif args.print_model_info:
        logger.info("Print model info mode selected")
        print_model_info()
    else:
        logger.info("No valid arguments provided, showing help")
        parser.print_help()
