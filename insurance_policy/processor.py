import asyncio
from collections.abc import Callable
from typing import Any

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm import get_embedding_model
from ragrock.extractor import Document, DocumentExtractor
from ragrock.pipeline import Pipeline
from ragrock.stage import (
    AsyncStage,
    DetectStage,
    Failed,
    ParallelExtractStage,
    ParallelStage,
)
from ragrock.stage.job import ExtractionJob
from utils import get_file_name, length_function
from vectordb import QdrantConnection


class ProcessingError(Exception):
    """Processing pieline exeception so it will raised if not able to process data."""

    def __init__(self, failures: list[Failed]):
        self.failures = failures
        details = "; ".join(f"{f.stage}: {f.error}" for f in failures)
        super().__init__(f"{len(failures)} document(s) failed to process. {details}")


# stages


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
            "text": doc.text,
        }


class LoadPdf(AsyncStage):
    async def process(self, job: ExtractionJob):
        doc = job.source
        if isinstance(doc, str) and (
            doc.startswith("https://") or doc.startswith("www")
        ):
            req = requests.get(doc)
            job.data = req.content
        elif isinstance(doc, bytes):
            job.data = doc
        else:
            with open(doc, "rb") as fp:
                job.data = fp.read()
        return job


class ParallelChunk(ParallelStage):
    def __init__(
        self,
        chunk_size=200,
        chunk_overlap=50,
        length_func: Callable | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_func

    def process(self, job: ExtractionJob):
        content = job.document.content()
        length_func = self.length_function or length_function
        splitter = RecursiveCharacterTextSplitter(
            length_function=length_func,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
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
                "point": await self.embedding_model.aembed_query(text),
            }

    async def process(self, job: ExtractionJob) -> ExtractionJob:
        doc = job.extra_metadata
        chunks = doc["chunks"]
        vec = await asyncio.gather(
            *(self.embed_doc(chunk, index) for index, chunk in enumerate(chunks))
        )
        job.extra_metadata["vector"] = vec
        return job


class VectorDB(AsyncStage):
    def __init__(self, user_id: str, **kwargs):
        super().__init__(**kwargs)
        self.user_id = user_id

    async def process(self, job: ExtractionJob) -> ExtractionJob:
        doc = job.extra_metadata
        vectors = doc["vector"]
        # Document-level metadata (e.g. doc_id) goes onto every chunk's payload.
        doc_meta = {k: v for k, v in doc.items() if k not in ("chunks", "vector")}
        async with QdrantConnection(user_id=self.user_id) as qdb:
            await qdb.add_point(vectors, metadata=doc_meta)

        return job


# public function


async def processor(
    sources: list[str],
    user_id: str,
    metadata: dict[str, dict] | None = None,
):
    """metadata: optional {source: {...}} stored on every chunk of that source."""
    metadata = metadata or {}
    extractor = DocumentExtractor()
    pipe = Pipeline(
        LoadPdf(workers=2),
        DetectStage(extractor.registry),
        ParallelExtractStage(extractor.registry, workers=2),
        ParallelChunk(chunk_size=1000, chunk_overlap=100),
        Vectorization(workers=8),
        VectorDB(user_id=user_id),
        threads=2,
        processes=2,
    )

    jobs = [
        ExtractionJob(
            file_name=get_file_name(source),
            source=source,
            file_type="pdf",
            extra_metadata=dict(metadata.get(source, {})),
        )
        for source in sources
    ]
    results = []
    failures: list[Failed] = []
    async for out in pipe.run(jobs):
        if isinstance(out, Failed):
            failures.append(out)
        else:
            results.append(out)

    if failures:
        raise ProcessingError(failures)

    return results
