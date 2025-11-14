import logging
import time

import torch

from .configs import model as model_config

logger = logging.getLogger(__name__)


def run_inference(model, decode_fn, max_new_tokens=500):
    context = torch.zeros((1, 1), dtype=torch.long, device=model_config.device)

    logger.info("Generating tokens...")
    start_time = time.time()
    out_tokens = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    generation_time = time.time() - start_time
    logger.info(f"Generated {len(out_tokens)} tokens in {generation_time:.2f}s")
    logger.debug(
        f"Token generation rate: {len(out_tokens) / generation_time:.1f} tokens/sec"
    )

    logger.debug("Decoding tokens to text...")
    text = decode_fn(out_tokens)
    logger.info(f"Text preview: {text[:100]}{'...' if len(text) > 100 else ''}")
    return text
