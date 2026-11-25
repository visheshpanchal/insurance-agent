"""Dummy backend for the Insurance Policy Assistant.

Every function here returns hard-coded placeholder data. Replace the bodies
with your real RAG pipeline (ingestion, retrieval, LLM, citation extraction)
without changing the signatures, and the UI in app.py keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass

from core import chat
from processor import processor
from utils import pdf_name

NOT_CONFIRMED = "I can't confirm this from the policy."


@dataclass
class Citation:
    document: str
    page: int
    section: str
    snippet: str


async def process_document(pdf_path: str, user_id: str) -> str:
    """Parse, chunk and embed a policy PDF. Returns a document id. (dummy)"""
    await processor([pdf_path], user_id)
    return pdf_name(pdf_path)


async def answer_question(doc_id: str, question: str, user_id: str):
    """Retrieve relevant chunks and answer with citations. (dummy)"""
    async for stream in chat(question, user_id):
        yield stream
