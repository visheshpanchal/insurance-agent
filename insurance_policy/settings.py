import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from qdrant_client.models import Distance

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


class EmbeddingModelSettings(BaseModel):
    embedding_model: str = "qwen/qwen3-embedding-4b"


class QdrantSettings(BaseModel):
    vector_size: int = 2560
    distance: Distance = Distance.COSINE
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY")
    qdrant_base_url: str = os.getenv("QDRANT_BASE_URL")


class ModelSettings(BaseModel):
    openrouter_api_key: str | None = os.getenv("OPENROUTER_KEY")
    openrouter_baseurl: str | None = os.getenv("OPENROUTER_URL")
    chat_model: str = "deepseek/deepseek-v4.1-flash-1017"


class Settings(EmbeddingModelSettings, QdrantSettings, ModelSettings):
    pass


settings = Settings()
