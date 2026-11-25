import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Filter,
    PointStruct,
    QueryResponse,
    SearchParams,
    VectorParams,
)
from settings import settings


def get_unique_id():
    return str(uuid.uuid4())


class QdrantConnection:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.client: AsyncQdrantClient | None = None

    def get_collection_config(self):
        return VectorParams(size=settings.vector_size, distance=settings.distance)

    async def __aenter__(self):
        self.client = AsyncQdrantClient(
            url=settings.qdrant_base_url, api_key=settings.qdrant_api_key
        )
        if not await self.client.collection_exists(self.user_id):
            await self.client.create_collection(
                self.user_id, vectors_config=self.get_collection_config()
            )
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.client:
            await self.client.close()

    async def check_connection(self):
        if not self.client:
            raise Exception("Connection is closed.")

    async def add_point(self, vectors: list, metadata: dict | None = None):
        if metadata is None:
            metadata = {}

        await self.client.upsert(
            collection_name=self.user_id,
            points=[
                PointStruct(
                    vector=vector["point"],
                    payload={"text": vector["text"], "metadata": metadata},
                    id=get_unique_id(),
                )
                for vector in vectors
            ],
        )

    async def delete_and_create(self):
        await self.client.delete(self.user_id)
        await self.client.create_collection(
            collection_name=self.user_id, vectors_config=self.get_collection_config()
        )

    async def query_points(
        self,
        query,
        query_filter: Filter | None = None,
        search_params: SearchParams | None = None,
        limit: int = 10,
        offset: int | None = None,
        with_payload: bool = False,
        **kwargs,
    ) -> QueryResponse:
        await self.check_connection()
        return await self.client.query_points(
            collection_name=self.user_id,
            query=query,
            query_filter=query_filter,
            search_params=search_params,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            **kwargs,
        )
