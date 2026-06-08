# src/utils/logging.py

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(
    experiment_name: str,
    dataset_name: str | None = None,
    log_root: str = "logs",
) -> logging.Logger:
    """
    Create a logger.

    Log path:
        logs/{dataset_name}/{experiment_name}/{timestamp}.log

    If dataset_name is not provided:
        logs/general/{experiment_name}/{timestamp}.log
    """
    dataset_name = dataset_name or "general"

    log_dir = Path(log_root) / dataset_name / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"{timestamp}.log"

    logger_name = f"{dataset_name}.{experiment_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Log file: {log_path}")

    return logger