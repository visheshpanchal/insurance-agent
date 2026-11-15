import os
import uuid

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, QueryResponse


def get_unique_id():
    return str(uuid.uuid4())


class QdrantSettings(BaseModel):
    collection_name: str = "default"
    vector_size: int = 512
    distance: Distance = Distance.COSINE
    url: str = "http://localhost:6333/"
    api_key: str = os.getenv("QDRANT_API_KEY", "your_secret_key")


class QdrantConnection:

    def __init__(self, qdrant_settings: QdrantSettings):
        self.settings = qdrant_settings
        self.client: AsyncQdrantClient | None = None

    def get_collection_config(self):
        return VectorParams(size=self.settings.vector_size,
                            distance=self.settings.distance)

    async def __aenter__(self):
        self.client = AsyncQdrantClient(url=self.settings.url, api_key=self.settings.api_key)
        if not await self.client.collection_exists(self.settings.collection_name):
            await self.client.create_collection(self.settings.collection_name,
                                                vectors_config=self.get_collection_config()
                                                )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.client:
            await self.client.close()

    async def check_connection(self):
        if not self.client:
            raise Exception("Connection is closed.")

    async def add_point(self, vectors: list, metadata: dict = None):
        if metadata is None:
            metadata = {}

        await self.client.upsert(
            collection_name=self.settings.collection_name,
            points=[PointStruct(
                vector=vector["point"],
                payload={"text": vector["text"], "metadata": metadata},
                id=get_unique_id()
            ) for vector in vectors]
        )

    async def delete_and_create(self):
        await self.client.delete(self.settings.collection_name)
        await self.client.create_collection(collection_name=self.settings.collection_name,
                                            vectors_config=self.get_collection_config())

    async def query_points(self, **kwargs) -> QueryResponse:
        await self.check_connection()
        return await self.client.query_points(collection_name=self.settings.collection_name ,**kwargs)
