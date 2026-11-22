"""Insurance Policy Assistant: Gradio UI.

Run:  python app.py
All backend calls go through services.py, which currently returns dummy data.
"""

from __future__ import annotations

import gradio as gr
import services
from memory import memory
from processor import ProcessingError
from ragrock.log import setup_logging
from services import Citation

setup_logging()

DISCLAIMER = (
    "Answers come only from the uploaded policy documents. "
    "This tool does not recommend which policy to buy."
)


def fmt_citation(c: Citation, n: int | None = None) -> str:
    label = f"**[{n}]** " if n is not None else ""
    return f"{label}`{c.document}` · p. {c.page} · *{c.section}*\n> {c.snippet}"


def clear_chat(request: gr.Request):
    user_id = request.session_hash
    memory.clear_history_chat(user_id)
    return [], "_No Citations_"


async def on_qa_upload(pdf, request: gr.Request):
    user_id = request.session_hash
    if not user_id:
        raise Exception("User id not available.")

    if pdf is None:
        return None, "No policy loaded.", []
    try:
        doc_id = await services.process_document(pdf, user_id)
    except ProcessingError as e:
        return None, f"Could not process the policy. {e}", []
    return doc_id, f"Loaded **{doc_id}**. Ask a question below.", []


# Feature: Q&A
async def on_ask(question, history, doc_id, request: gr.Request):
    history = history or []
    question = (question or "").strip()
    session_hash: str = request.session_hash
    if not session_hash:
        raise Exception("User id is not accessible.")
    if not question:
        yield "", history, gr.update()
        return

    if not doc_id:
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Please upload a policy PDF first."},
        ]
        yield "", history, "_No citations._"
        return

    history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ""},
    ]
    yield "", history, "_Generating answer…_"

    citations = []
    async for item in services.answer_question(doc_id, question, session_hash):
        if isinstance(item, str):
            history[-1]["content"] += item
            yield "", history, gr.update()
        else:
            citations = getattr(item, "citations", []) or []

    if citations:
        cites = "\n\n".join(fmt_citation(c, i) for i, c in enumerate(citations, 1))
    else:
        cites = "_No supporting passage found in the policy._"
    yield "", history, cites


# Gradio UI
with gr.Blocks(title="Insurance Policy Assistant") as demo:
    gr.Markdown(
        "# Insurance Policy Assistant\nCited answers from your insurance policy documents."
    )
    gr.Markdown(f"<sub>{DISCLAIMER}</sub>")

    with gr.Tabs():
        # Policy Q&A
        with gr.Tab("Policy Q&A"):
            doc_state = gr.State(None)
            with gr.Row():
                with gr.Column(scale=1):
                    qa_pdf = gr.File(
                        label="Policy PDF", file_types=[".pdf"], type="filepath"
                    )
                    qa_status = gr.Markdown("No policy loaded.")
                    gr.Markdown(
                        "**Try asking:**\n"
                        "- What is the waiting period for pre-existing diseases?\n"
                        "- Is there a cap on room rent?\n"
                        "- Does the policy cover dental treatment?"
                    )
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=420, label="Conversation")
                    with gr.Row():
                        question = gr.Textbox(
                            placeholder="Ask a question about your policy…",
                            show_label=False,
                            scale=5,
                        )
                        ask_btn = gr.Button("Ask", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear chat", size="sm")
                with gr.Column(scale=1):
                    gr.Markdown("### Citations")
                    qa_cites = gr.Markdown("_Citations appear here._")

            qa_pdf.change(on_qa_upload, [qa_pdf], [doc_state, qa_status, chatbot])
            ask_btn.click(
                on_ask, [question, chatbot, doc_state], [question, chatbot, qa_cites]
            )
            question.submit(
                on_ask, [question, chatbot, doc_state], [question, chatbot, qa_cites]
            )
            clear_btn.click(clear_chat, [], [chatbot, qa_cites])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), debug=True)
