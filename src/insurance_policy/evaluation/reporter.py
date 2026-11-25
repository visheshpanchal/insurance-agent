from loguru import logger
from ragrock.evaluation import (
    JsonlReporter,
    LogReporter,
    Reporter,
)

from evaluation.config import ANSWER_STAGE, EvalConfig


def print_summary(summary: dict) -> None:
    cols = ["n", "judge", "errors", "p50_latency_s"]
    header = f"{'group':<28}" + "".join(f"{c:>14}" for c in cols)

    def line(name, m):
        cells = "".join(f"{'-' if m[c] is None else m[c]:>14}" for c in cols)
        print(f"{name:<28}{cells}")

    print("\n" + header)
    print("-" * len(header))
    line("overall", summary["overall"])
    if summary["corpus_missing"]["n"]:
        line("(corpus missing)", summary["corpus_missing"])
    for key in ("by_category", "by_language", "by_difficulty", "by_expected_behavior"):
        print(f"\n[{key[3:]}]")
        for name, m in summary[key].items():
            line(f"  {name}", m)
    if summary["error_ids"]:
        print(f"\nErrors: {', '.join(summary['error_ids'])}")


class ConsoleReporter(Reporter):
    """One log line per question, and the breakdown table at the end."""

    def __init__(self, total: int | None = None):
        self.total = total
        self.done = 0

    def on_trace(self, trace) -> None:
        self.done += 1
        judge = (trace.evaluations.get("judge") or {}).get("score")
        stage = trace.stages.get(ANSWER_STAGE)
        error = trace.captured(ANSWER_STAGE, "error")
        logger.info(
            "[{}/{}] {} judge={} {:.1f}s{}",
            self.done,
            self.total or "?",
            trace.item_id,
            judge,
            stage.duration if stage else 0.0,
            f" ERROR {error}" if error else "",
        )

    def on_summary(self, summary) -> None:
        breakdown = summary.extra.get("breakdown")
        if breakdown:
            print_summary(breakdown)


def build_reporters(cfg: "EvalConfig", total: int | None) -> list[Reporter]:
    reporters: list[Reporter] = [LogReporter()] if cfg.log_stage_table else []
    return [*reporters, JsonlReporter(directory=cfg.runs_dir), ConsoleReporter(total)]
