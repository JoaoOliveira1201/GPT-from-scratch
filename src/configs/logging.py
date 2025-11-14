import logging
import os
from datetime import datetime

LOG_LEVEL = logging.DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL


def setup_logging():
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"logs/run-{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(LOG_LEVEL)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
