import logging
from logging.handlers import RotatingFileHandler

# Create log formatter
log_format = logging.Formatter(
    "[%(asctime)s.%(msecs)03d][%(threadName)s][%(levelname)s] %(message)s",
    datefmt="%Y.%m.%d-%H:%M:%S"
)

# Create root logger and set level
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Rotating file handler: rotate when file size exceeds 1GB, keep up to 20 backup files
file_handler = RotatingFileHandler(
    filename="logs/app.log",
    mode="a",
    maxBytes=1 * 1024 * 1024 * 512,  # 500MB
    backupCount=20,
    encoding="utf-8"
)
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

# Console handler: output to terminal
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)
logger.addHandler(console_handler)

# Example log
logger.info("Logger started.")
