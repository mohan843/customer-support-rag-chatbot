# Customer Support RAG Chatbot

A small, local retrieval-augmented customer-support chatbot that answers questions from a bundled FAQ knowledge base and cites the FAQ entry it used. It uses TF-IDF and cosine similarity, so the project runs without API keys, external services, or machine-learning model downloads.

## Description

The chatbot indexes each FAQ's question, answer, category, and keywords as a lightweight local corpus. At question time it ranks the FAQ entries by TF-IDF cosine similarity, returns the best answer with its stable FAQ citation, and falls back to a support contact when the FAQ does not contain a relevant answer. This is a compact, explainable RAG baseline rather than a generative LLM: responses are grounded in the source FAQ text.

## Project files

- `support_bot.py`: FAQ validation, TF-IDF retrieval, cited answer generation, and CLI.
- `data/faqs.json`: ten sample customer-support FAQs with paraphrase-friendly keywords.
- `tests/test_support_bot.py`: unit tests for retrieval, citations, fallbacks, and validation.
- `requirements.txt`: test dependency; runtime itself uses only the standard library.

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/mohan843/customer-support-rag-chatbot.git
cd customer-support-rag-chatbot
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage

Ask a question directly:

```bash
python support_bot.py "Where is my package right now?"
```

Example output:

```text
Open your order confirmation email and select Track order. Tracking can take up to 24 hours to appear after dispatch.

Source: [FAQ-001] Orders and shipping FAQ
```

Run interactively by omitting the question:

```bash
python support_bot.py
```

Use a different FAQ JSON file with `--faq path/to/faqs.json`. Each FAQ record needs a unique `id` and non-empty `category`, `question`, and `answer` fields; `keywords` is an optional list of strings.

Run tests:

```bash
python -m unittest discover -s tests -v
# or, after installing requirements:
pytest -q
```

## Extend it

Add entries to `data/faqs.json` to grow the knowledge base. For production use, replace the sample contact address and consider a privacy-reviewed embedding/vector database and a language model that is constrained to retrieved FAQ passages. Keep citations and the fallback behavior so unsupported answers remain transparent.
