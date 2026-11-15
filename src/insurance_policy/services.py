"""Dummy backend for the Insurance Policy Assistant.

Every function here returns hard-coded placeholder data. Replace the bodies
with your real RAG pipeline (ingestion, retrieval, LLM, citation extraction)
without changing the signatures, and the UI in app.py keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from processing_step import processor, Settings
from utils import pdf_name
import asyncio

NOT_CONFIRMED = "I can't confirm this from the policy."


@dataclass
class Citation:
    document: str
    page: int
    section: str
    snippet: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.text != NOT_CONFIRMED


@dataclass
class ChecklistResult:
    item: str
    found: bool
    detail: str
    citation: Citation | None = None


@dataclass
class ComparisonPoint:
    policy: str
    kind: str  # "pro" or "con"
    point: str
    citation: Citation


CHECKLIST_ITEMS = [
    "Waiting periods",
    "Room rent",
    "Co-payment",
    "Sub-limits",
    "Key exclusions",
    "Claim-notification timelines",
]


# ---------------------------------------------------------------------------
# 1. Policy Q&A
# ---------------------------------------------------------------------------
def process_document(pdf_path: str) -> str:
    """Parse, chunk and embed a policy PDF. Returns a document id. (dummy)"""
    settings = Settings()
    asyncio.run(processor([pdf_path], settings))
    return pdf_name(pdf_path)


def answer_question(doc_id: str, question: str) -> Answer:
    """Retrieve relevant chunks and answer with citations. (dummy)"""
    q = question.lower()
    if any(word in q for word in ("waiting", "pre-existing", "maternity", "room", "claim")):
        return Answer(
            text=(
                "Pre-existing diseases are covered after a waiting period of "
                "36 months of continuous coverage. [1]"
            ),
            citations=[
                Citation(
                    document=doc_id,
                    page=12,
                    section="4.2 Waiting Periods",
                    snippet=(
                        "Expenses related to pre-existing diseases shall be excluded "
                        "until the expiry of 36 months of continuous coverage..."
                    ),
                )
            ],
        )
    return Answer(text=NOT_CONFIRMED)


# ---------------------------------------------------------------------------
# 2. Policy checkup
# ---------------------------------------------------------------------------
def run_checkup(pdf_path: str) -> list[ChecklistResult]:
    """Check the policy against the fixed checklist. (dummy)"""
    doc = _name(pdf_path)
    return [
        ChecklistResult(
            "Waiting periods", True,
            "30 days initial; 24 months specific illnesses; 36 months pre-existing.",
            Citation(doc, 12, "4.2 Waiting Periods",
                     "A waiting period of 30 days shall apply from the policy start date..."),
        ),
        ChecklistResult(
            "Room rent", True,
            "Capped at 1% of sum insured per day (ICU 2%).",
            Citation(doc, 8, "3.1 In-patient Hospitalisation",
                     "Room, boarding and nursing expenses up to 1% of the sum insured per day..."),
        ),
        ChecklistResult(
            "Co-payment", True,
            "20% co-payment for insured persons aged 61 and above.",
            Citation(doc, 15, "5.3 Co-payment",
                     "A co-payment of 20% shall apply on each admissible claim where the insured is aged 61 years or above..."),
        ),
        ChecklistResult(
            "Sub-limits", False,
            "No sub-limit clause located in the policy.",
        ),
        ChecklistResult(
            "Key exclusions", True,
            "Cosmetic surgery, self-inflicted injury, war, experimental treatment.",
            Citation(doc, 18, "6. Exclusions",
                     "The company shall not be liable for any expenses arising from cosmetic or aesthetic treatment..."),
        ),
        ChecklistResult(
            "Claim-notification timelines", True,
            "Notify within 24 hours of emergency admission; 48 hours before planned admission.",
            Citation(doc, 21, "7.1 Claim Intimation",
                     "Intimation shall be given within 24 hours of admission in case of emergency hospitalisation..."),
        ),
    ]


# ---------------------------------------------------------------------------
# 3. Policy comparison
# ---------------------------------------------------------------------------
def compare_policies(pdf_paths: list[str], needs: str) -> list[ComparisonPoint]:
    """Compare policies against the user's needs; pros/cons only. (dummy)"""
    names = [_name(p) for p in pdf_paths]
    points: list[ComparisonPoint] = []
    samples = [
        ("pro", "No room-rent cap, which suits a preference for a private room.", 9, "3.1 Room Rent",
         "There shall be no capping on room category..."),
        ("con", "36-month waiting period for pre-existing conditions.", 12, "4.2 Waiting Periods",
         "Pre-existing diseases are covered after 36 months..."),
        ("pro", "Maternity covered after 9 months, with newborn cover included.", 14, "3.6 Maternity",
         "Maternity expenses are payable after a waiting period of 9 months..."),
        ("con", "20% co-payment on all claims regardless of age.", 16, "5.3 Co-payment",
         "A co-payment of 20% shall apply on every admissible claim..."),
        ("pro", "Restoration of sum insured once per policy year.", 10, "3.4 Restoration Benefit",
         "The sum insured shall be restored 100% once in a policy year..."),
        ("con", "Sub-limit of Rs 40,000 on cataract surgery per eye.", 11, "3.8 Sub-limits",
         "Cataract treatment is limited to Rs 40,000 per eye..."),
    ]
    for i, name in enumerate(names):
        for kind, point, page, section, snippet in (samples[i * 2 % 6], samples[(i * 2 + 1) % 6]):
            points.append(ComparisonPoint(name, kind, point, Citation(name, page, section, snippet)))
    return points
