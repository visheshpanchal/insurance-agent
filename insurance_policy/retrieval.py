from dataclasses import dataclass

from llm import get_embedding_model
from loguru import logger
from qdrant_client.models import Filter, QueryResponse, ScoredPoint, SearchParams
from ragrock.pipeline import Pipeline
from ragrock.stage import AsyncStage, Failed
from vectordb import QdrantConnection

QUERY_INSTRUCTION = "Given a question about an insurance policy, retrieve the policy clauses that answer it"


class RetrievalError(Exception):
    pass


@dataclass
class QueryRequest:
    """
    query - embedding vector of query to search data
    filter - Filter results retrieved from pipeline
    search - Search Params of query
    limit - limit of search results
    offset - offset of records
    score_threshold -
    """

    query: str | list[float | int]
    filter: Filter | None = None
    search: SearchParams | None = None
    limit: int = 10
    offset: int = 0
    score_threshold: float | None = None


class QueryEmbeddingStage(AsyncStage):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = get_embedding_model()

    async def process(self, query_request: QueryRequest):
        text = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query_request.query}"
        query_request.query = await self.embedding_model.aembed_query(text)
        return query_request


class RetrievalStage(AsyncStage):
    def __init__(self, user_id: str, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id

    async def process(self, request: QueryRequest) -> QueryResponse:
        if isinstance(request.query, str):
            raise ValueError(f"{request.query} should be vector.")

        async with QdrantConnection(self.user_id) as qdb:
            return await qdb.query_points(
                query=request.query,
                query_filter=request.filter,
                search_params=request.search,
                limit=request.limit,
                offset=request.offset,
                score_threshold=request.score_threshold,
                with_payload=True,
            )


async def retrieve_points(
    queries: list[str],
    user_id: str,
    top_k: int = 5,
    score_threshold: float = 0.4,
    limit_per_query: int = 10,
) -> list[ScoredPoint]:
    """
    Retriev points with ScoredPoint it helps to give more information for eval.
    """
    pipe = Pipeline(queue_size=10, threads=6, processes=2)

    pipe.add_stage(QueryEmbeddingStage(workers=2))
    pipe.add_stage(RetrievalStage(user_id=user_id, workers=2))

    jobs = [
        QueryRequest(
            query=query, limit=limit_per_query, score_threshold=score_threshold
        )
        for query in queries
    ]

    best: dict[str, ScoredPoint] = {}
    failures: list[Failed] = []
    async for result in pipe.run(jobs):
        if isinstance(result, Failed):
            failures.append(result)
            continue
        for point in result.points:
            key = str(point.id)
            if key not in best or point.score > best[key].score:
                best[key] = point

    if failures:
        logger.warning(
            "{} of {} retrieval queries failed: {}",
            len(failures),
            len(jobs),
            "; ".join(f"{f.stage}: {f.error!r}" for f in failures),
        )
        if len(failures) == len(jobs):
            raise RetrievalError(
                f"All retrieval queries failed. {failures[0].stage}: {failures[0].error!r}"
            )

    return sorted(best.values(), key=lambda p: p.score, reverse=True)[:top_k]


async def retrieve(
    queries: list[str],
    user_id: str,
    top_k: int = 5,
    score_threshold: float = 0.4,
    limit_per_query: int = 10,
) -> list[str]:
    points = await retrieve_points(
        queries, user_id, top_k, score_threshold, limit_per_query
    )

    out = []
    for point in points:
        if hasattr(point, "payload") and point.payload.get("get"):
            out.append(point.payload["text"])

    return out
