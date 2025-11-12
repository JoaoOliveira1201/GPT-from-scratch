import logging
import os

from .bpe import BPE
from ..configs import tokenizer as tokenizer_config
# Import logging config to initialize logging
from ..configs import logging as logging_config  # noqa: F401

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting tokenizer training process")
    bpe = BPE()

    combined_text = ""
    total_files = len(tokenizer_config.tokenizer_files)
    logger.info(f"Loading {total_files} training files for tokenizer")

    for i, file_path in enumerate(tokenizer_config.tokenizer_files, 1):
        try:
            logger.info(f"Reading file {i}/{total_files}: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
                combined_text += file_content
                logger.info(f"Loaded {len(file_content)} characters from {file_path}")
        except FileNotFoundError:
            logger.error(f"File '{file_path}' not found")
            raise
        except Exception as e:
            logger.error(f"Error reading file '{file_path}': {e}")
            raise

    if not combined_text:
        logger.error("No text content found in input files")
        exit(1)

    logger.info(f"Total combined text length: {len(combined_text)} characters")
    logger.info(f"Training tokenizer with vocab_size={tokenizer_config.tokenizer_vocab_size}")

    bpe.train(
        combined_text,
        tokenizer_config.tokenizer_vocab_size,
        tokenizer_config.tokenizer_verbose,
    )
    logger.info(f"Tokenizer training completed with {tokenizer_config.tokenizer_vocab_size} tokens")

    if not os.path.exists(tokenizer_config.tokenizer_output_dir):
        logger.info(f"Creating output directory: {tokenizer_config.tokenizer_output_dir}")
        os.makedirs(tokenizer_config.tokenizer_output_dir)

    output_path = os.path.join(
        tokenizer_config.tokenizer_output_dir, tokenizer_config.tokenizer_file_name
    )
    logger.info(f"Saving tokenizer to: {output_path}")
    bpe.save(output_path)
    logger.info("Tokenizer training and saving completed successfully")
