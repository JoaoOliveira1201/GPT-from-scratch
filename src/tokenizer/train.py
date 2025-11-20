import os

import src.logger as mlflow_logger

from ..configs import tokenizer as tokenizer_config
from .bpe import BPE

if __name__ == "__main__":
    mlflow_logger.info("Starting tokenizer training process")
    bpe = BPE()

    combined_text = ""
    total_files = len(tokenizer_config.tokenizer_files)

    for i, file_path in enumerate(tokenizer_config.tokenizer_files, 1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
                combined_text += file_content
                mlflow_logger.info(
                    f"Loaded {len(file_content)} characters from {file_path}"
                )
        except FileNotFoundError:
            mlflow_logger.error(f"File '{file_path}' not found")
            raise
        except Exception as e:
            mlflow_logger.error(f"Error reading file '{file_path}': {e}")
            raise

    if not combined_text:
        mlflow_logger.error("No text content found in input files")
        exit(1)

    mlflow_logger.info(f"Total files: {total_files}")
    mlflow_logger.info(f"Total characters: {len(combined_text)}")

    mlflow_logger.info(
        f"Training tokenizer with vocab_size={tokenizer_config.tokenizer_vocab_size}"
    )
    bpe.train(
        combined_text,
        tokenizer_config.tokenizer_vocab_size,
        tokenizer_config.tokenizer_verbose,
    )

    mlflow_logger.info(f"Number of merges: {len(bpe.merges)}")
    mlflow_logger.info(f"Final vocab size: {len(bpe.vocab)}")

    if not os.path.exists(tokenizer_config.tokenizer_output_dir):
        os.makedirs(tokenizer_config.tokenizer_output_dir)

    output_path = os.path.join(
        tokenizer_config.tokenizer_output_dir, tokenizer_config.tokenizer_file_name
    )
    bpe.save(output_path)
