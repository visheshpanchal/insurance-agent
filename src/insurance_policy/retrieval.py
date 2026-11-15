from ragrock.pipeline import Pipeline
from ragrock.stage import AsyncStage
from ragrock.stage.job import ExtractionJob
from dataclasses import dataclass
from qdrant_client.models import Filter, SearchParams
from .vectordb import QdrantConnection, QdrantSettings


@dataclass
class QueryRequest:
    """
    query - embedding vector of query to search data
    filter - Filter results retrieved from pipeline
    """
    query: list
    filter: Filter
    limit: int
    search: SearchParams

class RetrievalStage(AsyncStage):

    def __init__(self, settings: QdrantSettings):
        super().__init__()
        self.settings = QdrantSettings

    def process(self, request: QueryRequest):
        pass