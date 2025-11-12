import torch

# Model hyperparameters
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2
bias = False
block_size = 256

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Random seed
torch.manual_seed(1337)
