# Insurance Policy Assistant

A retrieval-augmented generation (RAG) app that answers questions about Indian health-insurance policy documents. Upload a policy PDF, ask a question in plain English (or Hinglish), and get an answer that walks through the relevant clauses. The answer is based only on the text of the document.

> Note: The assistant explains policies. It does not recommend which policy to buy. Every answer carries a note that it is AI-generated and should be checked with an insurance agent. This is just experimental project don't recommand to buy policy from this App.

## Features

- **PDF upload and indexing.** Each uploaded policy is chunked, embedded and stored in a per-session Qdrant collection.
- **Query enhancement.** An LLM rewrites the user's question into search-friendly queries, splits it into sub-queries when needed, and skips retrieval for small talk.
- **Multi-query retrieval.** Results from all sub-queries are merged, de-duplicated by best score, filtered by a score threshold and cut to the top-k.
- **Conversational memory.** Follow-up questions keep the chat history of the session.
- **Streaming answers** in a Gradio chat UI.
- **Evaluation harness.** An 84-question labelled dataset, an LLM-as-judge scorer, per-category breakdowns, and replay of saved runs.

## How it works

```
                 ┌──────────── Ingestion ─────────────┐
 Policy PDF ──►  LoadPdf ─► Chunk (1000 / 100 overlap) ─► Embed ─► Qdrant
                 └────────────────────────────────────┘

                 ┌──────────── Question ──────────────┐
 Question ──►    Query enhancer (LLM) ─► sub-queries ─► Embed ─► Qdrant search
                       │                                          │
                       │                       top-5 chunks (score ≥ 0.4)
                       ▼                                          ▼
                 chat history ───────────────►  Answer LLM (streamed) ─► UI
                 └────────────────────────────────────┘
```

| Component | Default |
|---|---|
| Chat model (via OpenRouter) | `deepseek/deepseek-v4.1-flash-1017` |
| Embedding model (via OpenRouter) | `qwen/qwen3-embedding-4b` (2560-dim, cosine) |
| Vector store | Qdrant (one collection per user session) |
| Pipeline framework | ragrock, for PDF extraction, staged async pipelines and eval |
| UI | Gradio 6 |

These defaults live in [`insurance_policy/settings.py`](insurance_policy/settings.py).

## Project structure

```
insurance-policy/
├── insurance_policy/
│   ├── app.py            # Gradio UI (entry point)
│   ├── services.py       # Thin layer the UI calls: process_document, answer_question
│   ├── core.py           # PolicyQANDA: query enhancement, context building, answer streaming
│   ├── processor.py      # Ingestion pipeline: load → chunk → embed → store
│   ├── retrieval.py      # Multi-query embedding + Qdrant search
│   ├── vectordb.py       # Async Qdrant connection helper
│   ├── memory.py         # In-process chat history per session
│   ├── llm.py            # Embedding model factory
│   ├── settings.py       # Models, Qdrant and OpenRouter config (reads .env)
│   ├── utils.py
│   ├── prompts/          # System prompts (reviewer, query enhancer, judge, analyzer)
│   └── evaluation/       # Eval harness: config, evaluators, reporter, entry point
├── data/                 # git-ignored
│   ├── pdf/              # Source documents for the eval corpus
│   ├── eval_dataset.jsonl
│   └── eval_runs/        # Saved run traces
├── Eval.md               # Eval results and roadmap
├── requirements.txt
└── .env.example
```

## Setup

### Prerequisites

- Python 3.12 or newer (the code uses `StrEnum` and PEP 604 types; development uses 3.13)
- A Qdrant instance (Qdrant Cloud, or local `docker run -p 6333:6333 qdrant/qdrant`)
- An [OpenRouter](https://openrouter.ai) API key
- The `ragrock` library. It is on PyPI

### Install

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure

Copy the example env file and fill it in:

```bash
cp .env.example .env
```

`settings.py` loads `.env` from the project root.

## Run the app

The modules use flat imports, so run them from inside the `insurance_policy/` directory:

```bash
cd insurance_policy
python app.py
```

Open the URL Gradio prints (by default http://127.0.0.1:7860). Then:

1. Upload a policy PDF in the **Policy PDF** box and wait for it to be indexed.
2. Ask a question, for example:
   - *What is the waiting period for pre-existing diseases?*
   - *Is there a cap on room rent?*
   - *Does the policy cover dental treatment?*
3. Use **Clear chat** to reset the conversation history.

## Evaluation

The eval harness checks the full pipeline (retrieval and answer) against `data/eval_dataset.jsonl`. Each row has a question, a ground-truth answer, key facts, reference document IDs and labels for `category`, `language`, `difficulty` and `expected_behavior` (`answer`, `answer_partial`, `correct_premise`, `refuse_out_of_scope`, `decline_harmful`, `flag_conflict`).

The corpus is a fixed set of public documents in `data/pdf/`, mapped in `evaluation/config.py`:

- Arogya Sanjeevani Policy, Go Digit (policy wording and CIS)
- IRDAI Master Circular on Health Insurance Business (2024)
- IRDAI Master Circular on Protection of Policyholders' Interests (2024)
- Insurance Ombudsman Rules, 2017 (as amended)
- IRDAI notice on premium hikes for senior citizens (2025)
- Income-tax Act 2025 (Section 80D / 126)

### Commands

There are no CLI flags. Edit the `EvalConfig(...)` call at the bottom of [`evaluation/main.py`](insurance_policy/evaluation/main.py), then run:

```bash
cd insurance_policy
python -m evaluation.main
```

| `command` | What it does |
|---|---|
| `Command.INGEST` | Drops the `eval_corpus` collection and re-indexes every PDF in `data/pdf/`. Run this first. |
| `Command.RUN` | Runs each question through the pipeline, records latency and retrieved chunks, and scores the answers if `judge=True`. Traces are saved to `data/eval_runs/`. |
| `Command.REPLAY` | Re-scores a saved run (`replay_path=...`) without calling retrieval or the answer model again. Use it to iterate on the judge prompt. |

Useful options: `num_questions` (seeded random sample), `seed`, `judge`, `judge_model` (defaults to the chat model) and `eval_workers`.

The judge scores each answer **0 / 1 / 2** (wrong / partial / correct and complete) against the reference answer and expected behaviour. Results are broken down by category, language, difficulty and expected behaviour.

### Current results

Baseline on a 5-question sample (see [`Eval.md`](Eval.md)):

| group | n | judge (0–2) | errors | p50 latency |
|---|---|---|---|---|
| overall | 5 | 2.0 | 0 | 22.8 s |
| factual_product | 2 | 2.0 | 0 | 18.3 s |
| factual_regulation | 2 | 2.0 | 0 | 36.3 s |
| multilingual (Hinglish) | 1 | 2.0 | 0 | 22.8 s |

A 5-question sample is a smoke test, not a benchmark. Run the full dataset before you compare changes.


## Disclaimer

This project is for learning and research. Its answers are AI-generated and may be wrong. They are not financial, legal or insurance advice. Always confirm with your insurer or a licensed agent.
