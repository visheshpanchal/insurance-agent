import os
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from vectordb import EmbeddingSetting

from dummy import DummyEmbeddings

api_key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1"


class ChatSettings(BaseModel):
    model: str = "deepseek/deepseek-v4.1-flash-1017"


# def get_embedding_model(settings: 'EmbeddingSetting'):
#     return OpenAIEmbeddings(api_key=api_key, model=settings.embedding_model,
#                             base_url=url, check_embedding_ctx_length=False,)

def get_embedding_model(settings: 'EmbeddingSetting'):
    return DummyEmbeddings()


