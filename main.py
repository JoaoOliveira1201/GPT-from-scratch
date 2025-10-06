from src.train import train_loop
from src.inference import run_inference


def main():
    model, decode = train_loop()
    text = run_inference(model, decode, max_new_tokens=500)
    print(text)


if __name__ == "__main__":
    main()
