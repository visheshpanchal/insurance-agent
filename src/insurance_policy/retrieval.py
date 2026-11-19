from dataclasses import dataclass

from qdrant_client.models import Filter, SearchParams
from ragrock.pipeline import Pipeline
from ragrock.stage import AsyncStage

from llm import get_embedding_model
from vectordb import QdrantConnection
from settings import settings


@dataclass
class QueryRequest:
    """
    query - embedding vector of query to search data
    filter - Filter results retrieved from pipeline
    """
    query: str | list[float | int]
    filter: Filter | None = None
    search: SearchParams | None = None
    limit: int = 10
    offset: int = 1


class QueryEmbeddingStage(AsyncStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = get_embedding_model()

    async def process(self, query_request: QueryRequest):
        out = await self.embedding_model.aembed_query(query_request.query)
        query_request.query = out
        return query_request


class RetrievalStage(AsyncStage):

    def __init__(self, user_id: str, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id

    async def process(self, request: QueryRequest):
        if isinstance(request, str):
            raise ValueError(f"{request.query} should be vector.")

        async with QdrantConnection(self.user_id) as qdb:
            out = await qdb.query_points(query=request.query,
                             query_filter=request.filter,
                             search_params=request.search,
                             limit=request.limit, offset=request.offset, with_payload=True)
            return out


async def retrieve(queries: list[str], user_id: str) -> list[str]:
    pipe = Pipeline(queue_size=10, threads=6, processes=2)

    pipe.add_stage(QueryEmbeddingStage(workers=2))
    pipe.add_stage(RetrievalStage(user_id=user_id, workers=2))

    jobs = [
        QueryRequest(query=query) for query in queries
    ]

    out = []
    async for result in pipe.run(jobs):
        out.append(result)

    return out
