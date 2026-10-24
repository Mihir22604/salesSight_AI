"""
Shared utilities for SalesSight AI:
- Structured logging (Console & File)
- Global reproducibility seed setting
- Domain-specific exceptions
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import Optional
import numpy as np


def get_logger(
    name: str = "salessight",
    log_file: Optional[str] = "logs/salessight.log",
    level: int = logging.INFO
) -> logging.Logger:
    """Returns a structured console and file logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """Enforces deterministic seeds across standard libraries and ML frameworks."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import sklearn
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# Domain Exceptions
class SalesSightException(Exception):
    """Base exception for SalesSight AI errors."""
    pass


class SchemaValidationError(SalesSightException):
    """Raised when incoming dataset schema violates required specifications."""
    pass


class DataIntegrityError(SalesSightException):
    """Raised when data transformations or consistency checks fail."""
    pass
