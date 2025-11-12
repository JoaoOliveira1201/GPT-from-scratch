import os

from .bpe import BPE
from ..configs import tokenizer as tokenizer_config

if __name__ == "__main__":
    bpe = BPE()

    combined_text = ""
    for file_path in tokenizer_config.tokenizer_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                combined_text += f.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found")
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}")

    if not combined_text:
        print("Error: No text content found in input files")
        exit(1)

    bpe.train(
        combined_text,
        tokenizer_config.tokenizer_vocab_size,
        tokenizer_config.tokenizer_verbose,
    )
    print(f"Tokenizer trained with {tokenizer_config.tokenizer_vocab_size} tokens")

    if not os.path.exists(tokenizer_config.tokenizer_output_dir):
        os.makedirs(tokenizer_config.tokenizer_output_dir)

    bpe.save(
        os.path.join(
            tokenizer_config.tokenizer_output_dir, tokenizer_config.tokenizer_file_name
        )
    )

    print(
        f"Tokenizer trained and saved to '{os.path.join(tokenizer_config.tokenizer_output_dir, tokenizer_config.tokenizer_file_name)}'"
    )
