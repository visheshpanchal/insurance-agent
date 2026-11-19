import asyncio
from typing import Any, Callable

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragrock.extractor import DocumentExtractor, Document
from ragrock.pipeline import Pipeline
from ragrock.stage import AsyncStage, DetectStage, ParallelExtractStage, ParallelStage
from ragrock.stage.job import ExtractionJob

from llm import get_embedding_model
from utils import get_file_name, length_function
from vectordb import QdrantConnection


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

    def __init__(self, control_request: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = get_embedding_model()
        self.control_request = control_request

    async def setup(self) -> None:
        self._semp = asyncio.Semaphore(self.control_request)

    async def embed_doc(self, text: str, index: int):
        async with self._semp:
            return {
                "index": index,
                "text": text,
                "point": await self.embedding_model.aembed_query(text)
            }

    async def process(self, job: ExtractionJob) -> ExtractionJob:
        doc = job.extra_metadata
        chunks = doc["chunks"]
        vec = await asyncio.gather(*(self.embed_doc(chunk, index) for index, chunk in enumerate(chunks)))
        job.extra_metadata["vector"] = vec
        return job


class VectorDB(AsyncStage):
    def __init__(self, user_id: str, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id

    async def process(self, job: ExtractionJob) -> ExtractionJob:
        doc = job.extra_metadata
        vectors = doc["vector"]
        async with QdrantConnection(user_id=self.user_id) as qdb:
            await qdb.add_point(vectors)

        return job


async def processor(sources: list[str], user_id: str):
    extractor = DocumentExtractor()
    pipe = Pipeline(
        LoadPdf(workers=2),
        DetectStage(extractor.registry),
        ParallelExtractStage(extractor.registry, workers=2),
        ParallelChunk(chunk_size=1000, chunk_overlap=100),
        Vectorization(workers=8),
        VectorDB(user_id=user_id),
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
