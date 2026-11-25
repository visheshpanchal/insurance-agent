import random
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from settings import settings

PKG_DIR = Path(__file__).resolve().parent.parent
ROOT = PKG_DIR.parents[0]
DEFAULT_COLLECTION = "eval_corpus"
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"
DATASET = DATA_DIR / "eval_dataset.jsonl"
RUNS_DIR = DATA_DIR / "eval_runs"

ANSWER_STAGE = "answer"
JUDGE_PROMPT_FILE = PKG_DIR / "prompts" / "eval-prompt.md"

# reference_doc_id -> file in data/pdf
DOC_FILES = {
    "AS_GODIGIT_PW": "policy-wordings-arogya-sanjeevani-policy-go-digit.pdf",
    "AS_GODIGIT_CIS": "CIS - Arogya Sanjeevani Policy, Go Digit.pdf",
    "MC_HEALTH_2024": "Master_Circular_on_Health_Insurance_Business_29052024.pdf",
    "PPHI_MC_2024": "MC_Protection_of_Policyholders_interests_2024.pdf",
    "OMB_RULES_2017": "Insurance Ombudsman Rules, 2017( as amended till 09.11.2023).pdf",
    "IRDAI_SENIOR_2025": "IRDAI Limits Health Insurance Premium Rate Hike for senior citizens.pdf",
    "IT_80D_126": "Income-tax-Act-2025.pdf",
}
# The CIS is a summary of the same product, so a CIS chunk counts as the policy wording.
DOC_ALIASES = {"AS_GODIGIT_CIS": "AS_GODIGIT_PW"}


class Command(StrEnum):
    """
    Ingest - Remove collections from database and reinsert datasets.
    Run - Just run score chat model.
    Replay - Rescore without running retrieval pipeline mostly good for evaluate llm-as-a-judge pipeline.
    """

    INGEST = "ingest"
    RUN = "run"
    REPLAY = "replay"


@dataclass
class EvalConfig:
    """Every option of one evaluation. Edit CONFIG below instead of passing arguments."""

    command: Command = Command.RUN
    collection: str = DEFAULT_COLLECTION

    num_questions: int | None = None
    seed: int = 42

    # Scoring (Command.RUN and Command.REPLAY)
    judge: bool = False
    judge_model: str | None = None
    eval_workers: int = 4  # questions scored at the same time (judge calls in parallel)
    replay_path: str | None = None  # e.g. "data/eval_runs/20261122T094623-4adba2.jsonl"

    # Output
    runs_dir: Path = RUNS_DIR
    log_stage_table: bool = True  # also log ragrock's stage timing table

    def __post_init__(self):
        self.command = Command(self.command)
        if self.eval_workers < 1:
            raise ValueError("eval_workers must be >= 1")
        if self.num_questions is not None and self.num_questions < 1:
            raise ValueError("num_questions must be >= 1 or None")
        if self.command is Command.REPLAY and not self.replay_path:
            raise ValueError("Command.REPLAY needs replay_path")
        self.runs_dir = Path(self.runs_dir)

    @property
    def judge_model_name(self) -> str | None:
        return (self.judge_model or settings.chat_model) if self.judge else None

    @property
    def replay_file(self) -> Path:
        path = Path(self.replay_path or "")
        return path if path.is_absolute() else ROOT / path

    def select(self, dataset: list[dict]) -> list[dict]:
        """Random sample of num_questions (seeded), kept in dataset order."""
        if self.num_questions is None or self.num_questions >= len(dataset):
            return list(dataset)
        picked = set(
            random.Random(self.seed).sample(range(len(dataset)), self.num_questions)
        )
        return [row for i, row in enumerate(dataset) if i in picked]

    def tags(self) -> dict[str, str]:
        """Stored with the run summary so runs can be compared later."""
        return {
            "command": str(self.command),
            "collection": self.collection,
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "judge_model": self.judge_model_name or "",
            "num_questions": str(self.num_questions or "all"),
            "seed": str(self.seed),
        }
