import logging
import os
import sys
from logging.handlers import RotatingFileHandler

_CONFIGURED = False
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "localtranscription.log")


def setup_logging(console_level=logging.WARNING, file_level=logging.INFO):
    """Configure root logging once. Idempotent across Flask reloader restarts.

    Defaults: console shows WARNING+ (program-breaking issues only),
    file keeps INFO+ for post-mortem debugging.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return logging.getLogger("localtranscription")

    os.makedirs(LOG_DIR, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(file_level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    stream_handler.setLevel(console_level)

    root = logging.getLogger()
    root.setLevel(min(file_level, console_level))
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Silence noisy third-party libraries on both handlers
    for noisy in ("werkzeug", "engineio", "socketio", "matplotlib",
                  "urllib3", "huggingface_hub", "transformers",
                  "nemo", "nemo.collections", "nemo_logger",
                  "torio", "torchaudio",
                  "pytorch_lightning", "lightning", "datasets",
                  "filelock", "fsspec", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    _CONFIGURED = True
    return logging.getLogger("localtranscription")


def get_logger(name):
    return logging.getLogger(name)
