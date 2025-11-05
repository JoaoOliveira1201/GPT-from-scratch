import os
import time
import torch

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2
train_test_split = 0.7
bias = False
model_path = os.path.join(os.path.dirname(__file__), "../weights/model.pth")
torch.manual_seed(1337)


def init_logger():
    import logging

    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"run-{timestamp}.log")

    logger = logging.getLogger("gpt_from_scratch")
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger, log_file
