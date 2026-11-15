"""Insurance Policy Assistant: Gradio UI.

Run:  python app.py
All backend calls go through services.py, which currently returns dummy data.
"""

from __future__ import annotations

import gradio as gr
import services
from services import Citation

DISCLAIMER = (
    "Answers come only from the uploaded policy documents. "
    "This tool does not recommend which policy to buy."
)


def fmt_citation(c: Citation, n: int | None = None) -> str:
    label = f"**[{n}]** " if n is not None else ""
    return f"{label}`{c.document}` · p. {c.page} · *{c.section}*\n> {c.snippet}"


def on_qa_upload(pdf):
    if pdf is None:
        return None, "No policy loaded.", []
    doc_id = services.process_document(pdf)
    return doc_id, f"✅ Loaded **{doc_id}**. Ask a question below.", []

# Feature: Q&A
def on_ask(question, history, doc_id):
    history = history or []
    question = (question or "").strip()
    if not question:
        return "", history, gr.update()
    if not doc_id:
        history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Please upload a policy PDF first."},
        ]
        return "", history, "_No citations._"

    ans = services.answer_question(doc_id, question)
    history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ans.text},
    ]
    if ans.citations:
        cites = "\n\n".join(fmt_citation(c, i) for i, c in enumerate(ans.citations, 1))
    else:
        cites = "_No supporting passage found in the policy._"
    return "", history, cites


# Feature: Policy checkup handlers
def on_checkup(pdf):
    if pdf is None:
        raise gr.Error("Upload a policy PDF first.")
    results = services.run_checkup(pdf)
    rows = [
        [
            r.item,
            "✅ Found" if r.found else "❌ Not found",
            r.detail,
            f"p. {r.citation.page}, {r.citation.section}" if r.citation else "—",
        ]
        for r in results
    ]
    found = sum(r.found for r in results)
    summary = f"**{found} / {len(results)}** checklist items found in the policy."
    cites = (
        "\n\n".join(
            f"#### {r.item}\n{fmt_citation(r.citation)}" for r in results if r.citation
        )
        or "_No citations._"
    )
    return summary, rows, cites


# Feature: Policy comparison handlers
def on_compare(pdf_a, pdf_b, pdf_c, needs):
    pdfs = [p for p in (pdf_a, pdf_b, pdf_c) if p is not None]
    # TODO: Change we can compare with 2 PDFs
    if len(pdfs) != 3:
        raise gr.Error("Upload all 3 policy PDFs.")
    if not (needs or "").strip():
        raise gr.Error("Describe your needs first.")

    points = services.compare_policies(pdfs, needs)
    outputs = []
    for pdf in pdfs:
        name = services._name(pdf)
        mine = [p for p in points if p.policy == name]
        md = [f"### {name}"]
        for kind, title in (("pro", "👍 Pros"), ("con", "👎 Cons")):
            md.append(f"**{title}**")
            items = [p for p in mine if p.kind == kind]
            if not items:
                md.append("- _None found._")
            for p in items:
                c = p.citation
                md.append(
                    f"- {p.point}  \n  <sub>`{c.document}` · p. {c.page} · {c.section}</sub>"
                )
        outputs.append("\n".join(md))
    return (*outputs, f"_Compared against your needs: “{needs.strip()}”. {DISCLAIMER}_")


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

            qa_pdf.change(on_qa_upload, qa_pdf, [doc_state, qa_status, chatbot])
            ask_btn.click(
                on_ask, [question, chatbot, doc_state], [question, chatbot, qa_cites]
            )
            question.submit(
                on_ask, [question, chatbot, doc_state], [question, chatbot, qa_cites]
            )
            clear_btn.click(
                lambda: ([], "_Citations appear here._"), None, [chatbot, qa_cites]
            )

        # Policy Checkup
        with gr.Tab("Policy Checkup"):
            gr.Markdown(
                "Checks the policy for: "
                + ", ".join(f"**{i}**" for i in services.CHECKLIST_ITEMS)
                + "."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    ck_pdf = gr.File(
                        label="Policy PDF", file_types=[".pdf"], type="filepath"
                    )
                    ck_btn = gr.Button("Run checkup", variant="primary")
                    ck_summary = gr.Markdown()
                with gr.Column(scale=3):
                    ck_table = gr.Dataframe(
                        headers=[
                            "Checklist item",
                            "Status",
                            "What the policy says",
                            "Citation",
                        ],
                        datatype=["str", "str", "str", "str"],
                        value=[[i, "—", "", ""] for i in services.CHECKLIST_ITEMS],
                        interactive=False,
                        wrap=True,
                    )
            with gr.Accordion("Cited passages", open=False):
                ck_cites = gr.Markdown("_Run a checkup to see cited passages._")

            ck_btn.click(on_checkup, ck_pdf, [ck_summary, ck_table, ck_cites])

        # Policy Comparison
        with gr.Tab("Policy Comparison"):
            gr.Markdown(
                "Upload 3 policies and describe what matters to you. "
                "You get cited pros and cons for each, **not a recommendation**."
            )
            with gr.Row():
                cmp_a = gr.File(label="Policy A", file_types=[".pdf"], type="filepath")
                cmp_b = gr.File(label="Policy B", file_types=[".pdf"], type="filepath")
                cmp_c = gr.File(label="Policy C", file_types=[".pdf"], type="filepath")
            needs = gr.Textbox(
                label="Your needs",
                lines=3,
                placeholder="e.g. Family of 4, parents aged 65, want maternity cover, prefer a private room, low co-pay.",
            )
            cmp_btn = gr.Button("Compare policies", variant="primary")
            with gr.Row(equal_height=True):
                out_a = gr.Markdown()
                out_b = gr.Markdown()
                out_c = gr.Markdown()
            cmp_note = gr.Markdown()

            cmp_btn.click(
                on_compare,
                [cmp_a, cmp_b, cmp_c, needs],
                [out_a, out_b, out_c, cmp_note],
            )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), debug=True)
