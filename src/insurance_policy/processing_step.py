import asyncio
import io
from typing import Any, Callable

import requests
from pydantic import BaseModel
from ragrock.extractor import DocumentExtractor, Document
from ragrock.pipeline import Pipeline
from ragrock.stage import AsyncStage, DetectStage, ParallelExtractStage, ParallelStage
from ragrock.stage.job import ExtractionJob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vectordb import QdrantSettings, QdrantConnection
from dummy import dummy_embeddings

from utils import get_file_name, length_function, get_embedding_model

class Settings(BaseModel):
    embedding_model: str = "nvidia/nemotron-3.5-embed-8b:free"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api: str = "your_secret_key"

qdb_settings = QdrantSettings(
    vector_size=4096,
    collection_name="default"
)

class PreviewStage(AsyncStage):

    async def process(self, job: ExtractionJob) -> dict[str, str | None | int | Any]:
        doc = job.document
        if not isinstance(doc, Document):
            return {"error": "Document is not a Document"}

        return {
            "name": job.file_name,
            "pages": doc.metadata.page_count,
            "words": doc.word_count,
            "tables": len(doc.tables),
            "text": doc.text
        }


class LoadPdf(AsyncStage):
    async def process(self, job: ExtractionJob):
        doc = job.source
        if isinstance(doc, str) and (doc.startswith("https://") or doc.startswith("www")):
            req = requests.get(doc)
            job.data = req.content
        elif isinstance(doc, bytes):
            job.data = doc
        else:
            with open(doc, "rb") as fp:
                job.data = fp.read()
        return job


class ParallelChunk(ParallelStage):
    def __init__(self, chunk_size=200, chunk_overlap=50, length_func: Callable | None = None, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_func

    def process(self, job: ExtractionJob):
        content = job.document.content()
        length_func = self.length_function or length_function
        splitter = RecursiveCharacterTextSplitter(length_function=length_func,
                                                  chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = splitter.split_text(content)

        if not isinstance(job.extra_metadata, dict):
            job.extra_metadata = {}

        job.extra_metadata["chunks"] = docs.copy()

        return job

class Vectorization(AsyncStage):

    def __init__(self, settings: Settings, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings
        self.embedding_model = get_embedding_model(settings)

    async def process(self, job: ExtractionJob) -> ExtractionJob:
        doc = job.extra_metadata
        chunks = doc["chunks"]
        vec = []
        for chunk in chunks:
            point = await self.embedding_model.aembed_query(chunk)

            vec.append({'text': chunk, 'point': point})

        job.extra_metadata["vector"] = vec
        return job

class VectorDB(AsyncStage):
    def __init__(self, settings: Settings, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings

    async def process(self, job: ExtractionJob) -> ExtractionJob:
        doc = job.extra_metadata
        vectors = doc["vector"]
        async with QdrantConnection(qdrant_settings=qdb_settings) as qdb:
            await qdb.add_point(vectors)

        return job

async def processor(sources: list, settings: Settings):
    extractor = DocumentExtractor()
    pipe = Pipeline(
        LoadPdf(workers=2),
        DetectStage(extractor.registry),
        ParallelExtractStage(extractor.registry, workers=2),
        ParallelChunk(chunk_size=500, chunk_overlap=100),
        Vectorization(settings=settings, workers=8),
        VectorDB(settings=qdb_settings),
        # PreviewStage(),
        threads=2,
        processes=2
    )

    jobs = [ExtractionJob(file_name=get_file_name(source),
                          source=source,
                          file_type="pdf")
            for source in sources]
    results = []
    async for out in pipe.run(jobs):
        results.append(out)

    return results


