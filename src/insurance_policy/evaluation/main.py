from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from loguru import logger
from ragrock.evaluation import (
    EvalOptions,
    Labeled,
    areplay,
    capture,
    record,
)
from ragrock.pipeline import Pipeline
from ragrock.stage import AsyncStage

from core import PolicyQANDA
from evaluation.config import DOC_FILES, PDF_DIR, Command, EvalConfig, DOC_ALIASES, ANSWER_STAGE, DATASET
from evaluation.evaluators import build_evaluators, Breakdown, corpus_missing
from evaluation.reporter import build_reporters
from memory import memory
from processor import processor
from vectordb import QdrantConnection


def load_dataset(path: Path = DATASET) -> list[dict]:
    with open(path, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]





async def ingest(collection: str) -> None:
    sources, meta = [], {}
    for doc_id, name in DOC_FILES.items():
        path = PDF_DIR / name
        if not path.exists():
            logger.warning("Missing PDF for {}: {}", doc_id, path)
            continue
        sources.append(str(path))
        meta[str(path)] = {"doc_id": doc_id, "file": name}

    # Start from an empty collection so re-ingesting doesn't duplicate chunks.
    async with QdrantConnection(collection) as qdb:
        await qdb.client.delete_collection(collection)

    logger.info("Ingesting {} PDFs into collection '{}'", len(sources), collection)
    await processor(sources, collection, metadata=meta)

    async with QdrantConnection(collection) as qdb:
        count = await qdb.client.count(collection)
    logger.info("Done: {} chunks indexed", count.count)

    missing = sorted({d for row in load_dataset() for d in corpus_missing(row)})
    if missing:
        logger.warning(
            "Dataset references docs with no PDF (their questions are reported separately): {}",
            missing,
        )


def doc_id_of(point) -> str | None:
    meta = (point.payload or {}).get("metadata") or {}
    doc_id = meta.get("doc_id")
    return DOC_ALIASES.get(doc_id, doc_id)


class AnswerStage(AsyncStage):
    """Runs the real chat pipeline (query rewrite -> retrieval -> LLM) for one question."""

    def __init__(self, collection: str):
        # workers=1: memory is a process-wide dict keyed by the collection id.
        super().__init__(name=ANSWER_STAGE, workers=1)
        self.collection = collection

    async def process(self, question: str) -> dict:
        # Fresh chat history per question; the collection (same id) stays as ingested.
        memory.clear_history_chat(self.collection)
        bot = PolicyQANDA(user_id=self.collection)

        error = None
        try:
            answer = "".join([token async for token in bot.chat(question)])
        except Exception as e:  # noqa: BLE001 - recorded, scored as a failed answer
            logger.exception("Chat pipeline failed")
            answer, error = "", repr(e)

        retrieved = [doc_id_of(p) for p in bot.last_points]

        record("error", error is not None)
        record("chunks", len(bot.last_points))
        record("answer_chars", len(answer))
        capture("answer", answer)
        capture("retrieved_docs", retrieved)
        capture("error", error)
        return {"answer": answer, "retrieved_docs": retrieved, "error": error}


async def rescore(config: EvalConfig) -> None:
    """Re-run the evaluators on a saved run; the chat pipeline is not called."""
    path = config.replay_file
    if not path.exists():
        raise SystemExit(f"Run file not found: {path}")
    rows_by_id = {r["id"]: r for r in load_dataset()}
    await areplay(
        path,
        evaluators=build_evaluators(config, rows_by_id),
        run_evaluators=[Breakdown(rows_by_id)],
        reporters=build_reporters(config, None),
        workers=config.eval_workers,
        tags=config.tags(),
    )


async def run(config: EvalConfig) -> None:
    """Run process of evalution"""
    dataset = load_dataset()
    rows_by_id = {r["id"]: r for r in dataset}
    rows = config.select(dataset)
    if not rows:
        raise SystemExit("No questions selected; check the filters in CONFIG.")

    # pipeline for AnswerStage
    pipe = Pipeline(
        AnswerStage(config.collection),
        queue_size=4,
        threads=4,
        evaluators=build_evaluators(config, rows_by_id),
        run_evaluators=[Breakdown(rows_by_id)],
        reporters=build_reporters(config, len(rows)),
    )
    cases = [Labeled(r["question"], labels=r, id=r["id"]) for r in rows]
    opts = EvalOptions(evaluate=True, workers=config.eval_workers, tags=config.tags())

    logger.info("Evaluating {} questions (runs -> {})", len(rows), config.runs_dir)
    async for _ in pipe.run(cases, evaluation=opts):
        pass


def main(config: EvalConfig):
    os.environ.pop("RAGROCK_EVAL", None)
    if config.command is Command.INGEST:
        # Only Ingest data into vectordb
        asyncio.run(ingest(config.collection))
    elif config.command is Command.RUN:
        # Use PolicyQ&A pipeline retrieve information then calculate metrics
        asyncio.run(run(config))
    else:
        # This won't be call again retrieve pipeline mostly used for LLM-as-Judge
        asyncio.run(rescore(config))


if __name__ == "__main__":
    config = EvalConfig(num_questions=5, seed=42, judge=True, command=Command.REPLAY, replay_path="data/eval_runs/20261125T120807-9a84ab.jsonl")
    main(config)
