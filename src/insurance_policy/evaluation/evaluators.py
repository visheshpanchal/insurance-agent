import statistics
from collections import defaultdict

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from ragrock.evaluation import (
    Evaluator,
    RunEvaluator,
)

from evaluation.config import JUDGE_PROMPT_FILE, ANSWER_STAGE, DOC_FILES
from settings import settings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.main import EvalConfig


class DatasetLabelsMixin:
    """Reads the row from the current dataset, so `replay` picks up regex fixes.

    Falls back to the labels saved in the trace if the id is no longer in the dataset.
    """

    def __init__(self, rows: dict[str, dict] | None = None):
        self.rows = rows or {}

    def row(self, trace) -> dict:
        return self.rows.get(trace.item_id) or trace.labels


def corpus_missing(row: dict) -> list[str]:
    return sorted(set(row["reference_doc_ids"]) - set(DOC_FILES))


def judge_system_prompt() -> str:
    text = JUDGE_PROMPT_FILE.read_text(encoding="utf-8")
    return text.replace("{", "{{").replace("}", "}}")


def mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(statistics.mean(values), 3) if values else None


class JudgeScorer(DatasetLabelsMixin, Evaluator):
    """LLM-as-judge: 0 / 1 / 2 against the reference answer (prompts/eval-prompt.md)."""

    name = "judge"
    requires = {ANSWER_STAGE: ["answer"]}  # noqa: RUF012

    def __init__(self, model_name: str, rows: dict[str, dict] | None = None):
        super().__init__(rows)
        self.model_name = model_name
        self.chain = None

    def setup(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", judge_system_prompt()),
                (
                    "user",
                    (
                        "Question:\n{question}\n\nExpected behaviour: {behavior}\n\n"
                        "Reference answer:\n{ground_truth}\n\nAssistant answer:\n{answer}"
                    ),
                ),
            ]
        )
        model = ChatOpenAI(
            model=self.model_name,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_baseurl,
            temperature=0,
        )
        self.chain = prompt | model | JsonOutputParser()

    async def evaluate(self, item, trace):
        if trace.captured(ANSWER_STAGE, "error"):
            return None  # nothing to judge; counted under "errors" in the breakdown
        row = self.row(trace)
        out = await self.chain.ainvoke(
            {
                "question": row["question"],
                "behavior": row["expected_behavior"],
                "ground_truth": row["ground_truth"],
                "answer": trace.captured(ANSWER_STAGE, "answer") or "(empty)",
            }
        )
        # Errors here are logged by ragrock and counted in summary.eval_errors.
        return {"score": int(out["score"]), "detail": {"reason": out.get("reason")}}


def summarize(results: list[dict]) -> dict:
    def metrics(rows: list[dict]) -> dict:
        latencies = [r["latency_s"] for r in rows if r["latency_s"] is not None]
        return {
            "n": len(rows),
            "judge": mean(r["judge_score"] for r in rows),
            "errors": sum(1 for r in rows if r["error"]),
            "p50_latency_s": round(statistics.median(latencies), 2)
            if latencies
            else None,
        }

    complete = [r for r in results if not r["corpus_missing"]]
    summary = {
        "overall": metrics(complete),
        "corpus_missing": metrics([r for r in results if r["corpus_missing"]]),
    }
    for group in ("category", "language", "difficulty", "expected_behavior"):
        groups = defaultdict(list)
        for r in complete:
            groups[r[group]].append(r)
        summary[f"by_{group}"] = {k: metrics(v) for k, v in sorted(groups.items())}
    summary["error_ids"] = sorted(r["id"] for r in results if r["error"])
    return summary


class Breakdown(DatasetLabelsMixin, RunEvaluator):
    """Per-group metrics; full table in summary.extra["breakdown"]."""

    name = "breakdown"
    needs_traces = True

    def evaluate(self, summary, traces):
        results = []
        for t in traces or []:
            row = self.row(t)
            stage = t.stages.get(ANSWER_STAGE)
            results.append(
                {
                    "id": t.item_id,
                    "category": row["category"],
                    "language": row["language"],
                    "difficulty": row["difficulty"],
                    "expected_behavior": row["expected_behavior"],
                    "corpus_missing": corpus_missing(row),
                    "error": t.captured(ANSWER_STAGE, "error")
                             or (stage.error if stage else "no answer stage"),
                    "latency_s": stage.duration if stage else None,
                    "judge_score": (t.evaluations.get("judge") or {}).get("score"),
                }
            )
        breakdown = summarize(results)
        summary.extra["breakdown"] = breakdown
        return {k: breakdown["overall"][k] for k in ("n", "judge", "errors")}


def build_evaluators(config: "EvalConfig", rows_by_id: dict[str, dict]) -> list[Evaluator]:
    evaluators: list[Evaluator] = []  # the rule scorer gets added back here later
    if config.judge:
        evaluators.append(JudgeScorer(config.judge_model_name, rows_by_id))
    return evaluators
