import torch
from . import config


def run_inference(model, decode_fn, max_new_tokens=500):
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    out_tokens = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    return decode_fn(out_tokens)


