from src.train import train_loop
from src.inference import run_inference
from src.model import GPTLanguageModel
from src import config

def train_and_inference():
    model, decode = train_loop()
    text = run_inference(model, decode, max_new_tokens=500)
    print(text)


def print_model_info():
    model = GPTLanguageModel()
    print(model)
    print(f"parameters={sum(p.numel() for p in model.parameters())/1e6:.3f}M, device={config.device}")


if __name__ == "__main__":
    train_and_inference()
