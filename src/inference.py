import logging
import time
import torch
from .configs import model as model_config

logger = logging.getLogger(__name__)


def run_inference(model, decode_fn, max_new_tokens=500):
    logger.info(f"Starting inference with max_new_tokens={max_new_tokens}")

    context = torch.zeros((1, 1), dtype=torch.long, device=model_config.device)
    logger.debug(f"Initial context shape: {context.shape}, device: {context.device}")

    logger.info("Generating tokens...")
    start_time = time.time()
    out_tokens = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    generation_time = time.time() - start_time
    logger.info(f"Generated {len(out_tokens)} tokens in {generation_time:.2f}s")
    logger.debug(f"Token generation rate: {len(out_tokens)/generation_time:.1f} tokens/sec")

    logger.debug("Decoding tokens to text...")
    decode_start = time.time()
    text = decode_fn(out_tokens)
    decode_time = time.time() - decode_start
    logger.info(f"Decoded text length: {len(text)} characters in {decode_time:.2f}s")
    logger.debug(f"Text preview: {text[:100]}{'...' if len(text) > 100 else ''}")
    return text
