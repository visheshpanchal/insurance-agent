from pathlib import Path

from loguru import logger


def get_file_name(file: str):
    logger.warning("Implementation is left for get_file_name function.")


def pdf_name(pdf_path: str | None) -> str:
    return Path(pdf_path).name if pdf_path else "policy.pdf"


def length_function(text: str):
    return len(text) // 4
