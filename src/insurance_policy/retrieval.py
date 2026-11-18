from dataclasses import dataclass

from qdrant_client.models import Filter, SearchParams
from ragrock.pipeline import Pipeline
from ragrock.stage import AsyncStage

from llm import get_embedding_model
from vectordb import QdrantConnection, QdrantSettings


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
    def __init__(self, settings, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings
        self.embedding_model = get_embedding_model(self.settings)

    async def process(self, query_request: QueryRequest):
        out = await self.embedding_model.aembed_query(query_request.query)
        query_request.query = out
        return query_request


class RetrievalStage(AsyncStage):

    def __init__(self, settings: QdrantSettings, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings

    async def process(self, request: QueryRequest):
        if isinstance(request, str):
            raise ValueError(f"{request.query} should be vector.")

        async with QdrantConnection(self.settings) as qdb:
            out = await qdb.query_points(query=request.query,
                             query_filter=request.filter,
                             search_params=request.search,
                             limit=request.limit, offset=request.offset, with_payload=True)
            return out


async def retrieve(queries: list[str], eval=False) -> list[str]:
    pipe = Pipeline(queue_size=10, threads=6, processes=2)
    qdb_settings = QdrantSettings(
        vector_size=4096,
        collection_name="default"  # uuid.uuid4()
    )
    pipe.add_stage(QueryEmbeddingStage(settings=qdb_settings, workers=2))
    pipe.add_stage(RetrievalStage(settings=qdb_settings, workers=2))

    jobs = [
        QueryRequest(query=query) for query in queries
    ]

    out = []
    async for result in pipe.run(jobs):
        out.append(result)

    return out
