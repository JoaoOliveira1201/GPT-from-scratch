import logging
import torch
from .configs import model as model_config

logger = logging.getLogger(__name__)


def run_inference(model, decode_fn, max_new_tokens=500):
    logger.info(f"Starting inference with max_new_tokens={max_new_tokens}")

    context = torch.zeros((1, 1), dtype=torch.long, device=model_config.device)
    logger.info("Generating tokens...")
    out_tokens = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    logger.info(f"Generated {len(out_tokens)} tokens")

    text = decode_fn(out_tokens)
    logger.info(f"Decoded text length: {len(text)} characters")
    return text
