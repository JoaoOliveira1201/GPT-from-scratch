import logging
import os
from datetime import datetime

# Configure logging with environment variable support
log_level = os.getenv('GPT_LOG_LEVEL', 'INFO').upper()
level_map = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Create a timestamped log file
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_file = f"logs/run-{timestamp}.log"

# Create logger
logger = logging.getLogger()
logger.setLevel(level_map.get(log_level, logging.INFO))

# Remove any existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(level_map.get(log_level, logging.INFO))
console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(console_formatter)

# Create file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(level_map.get(log_level, logging.INFO))
file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Suppress some noisy loggers
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)
