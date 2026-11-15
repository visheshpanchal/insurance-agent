from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from dummy import DummyEmbeddings

if TYPE_CHECKING:
    from processing_step import Settings


def get_file_name(file: str):
    logger.warning("Implementation is left for get_file_name function.")


def pdf_name(pdf_path: str | None) -> str:
    return Path(pdf_path).name if pdf_path else "policy.pdf"


def length_function(text: str):
    return len(text) // 4


# def get_embedding_model(settings: 'Settings'):
#     return OpenAIEmbeddings(api_key=os.getenv("OPENROUTER_API_KEY"), model=settings.embedding_model,
#                             base_url="https://openrouter.ai/api/v1", check_embedding_ctx_length=False,)

def get_embedding_model(settings: 'Settings'):
    return DummyEmbeddings()


