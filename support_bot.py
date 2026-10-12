"""A tiny, dependency-light FAQ RAG chatbot using TF-IDF retrieval."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_FAQ_PATH = Path(__file__).parent / "data" / "faqs.json"
FALLBACK_ANSWER = (
    "I couldn't find an answer in the FAQ. Please contact support@example.com "
    "and include your order number if you have one."
)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or",
    "our", "the", "to", "we", "what", "when", "where", "which", "with",
    "you", "your",
}


@dataclass(frozen=True)
class FAQ:
    """One support answer and its citation metadata."""

    id: str
    category: str
    question: str
    answer: str
    keywords: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, item: dict[str, object]) -> "FAQ":
        required = ("id", "category", "question", "answer")
        missing = [field for field in required if not str(item.get(field, "")).strip()]
        if missing:
            raise ValueError(f"FAQ entry is missing required field(s): {', '.join(missing)}")
        keywords = item.get("keywords", [])
        if not isinstance(keywords, list) or not all(isinstance(word, str) for word in keywords):
            raise ValueError(f"FAQ {item['id']} keywords must be a list of strings")
        return cls(
            id=str(item["id"]).strip(),
            category=str(item["category"]).strip(),
            question=str(item["question"]).strip(),
            answer=str(item["answer"]).strip(),
            keywords=tuple(keywords),
        )

    def searchable_text(self) -> str:
        return " ".join((self.question, self.answer, self.category, *self.keywords))


@dataclass(frozen=True)
class SearchResult:
    faq: FAQ
    score: float


def tokenize(text: str) -> list[str]:
    """Lowercase and remove common stop words while preserving useful terms."""
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token.lower() not in STOP_WORDS and len(token) > 1
    ]


def load_faqs(path: str | Path = DEFAULT_FAQ_PATH) -> list[FAQ]:
    """Load and validate a JSON array of FAQ records."""
    with Path(path).open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, list) or not raw:
        raise ValueError("FAQ file must contain a non-empty JSON array")
    faqs = [FAQ.from_dict(item) for item in raw if isinstance(item, dict)]
    if len(faqs) != len(raw):
        raise ValueError("Every FAQ entry must be a JSON object")
    ids = [faq.id for faq in faqs]
    if len(ids) != len(set(ids)):
        raise ValueError("FAQ IDs must be unique")
    return faqs


class FAQRetriever:
    """Rank FAQ entries using local TF-IDF cosine similarity (no model/API needed)."""

    def __init__(self, faqs: Iterable[FAQ]):
        self.faqs = list(faqs)
        if not self.faqs:
            raise ValueError("At least one FAQ is required")
        ids = [faq.id for faq in self.faqs]
        if len(ids) != len(set(ids)):
            raise ValueError("FAQ IDs must be unique")
        documents = [tokenize(faq.searchable_text()) for faq in self.faqs]
        document_frequency: dict[str, int] = {}
        for terms in documents:
            for term in set(terms):
                document_frequency[term] = document_frequency.get(term, 0) + 1
        self.idf = {
            term: math.log(1 + len(documents) / frequency) + 1
            for term, frequency in document_frequency.items()
        }
        self.vectors = [self._vector(terms) for terms in documents]

    def _vector(self, terms: list[str]) -> dict[str, float]:
        counts: dict[str, int] = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        vector = {
            term: (1 + math.log(count)) * self.idf.get(term, 0.0)
            for term, count in counts.items()
            if term in self.idf
        }
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        return {term: weight / norm for term, weight in vector.items()} if norm else {}

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Return up to top_k most similar FAQs, highest score first."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_vector = self._vector(tokenize(query))
        if not query_vector:
            return []
        results = [
            SearchResult(
                faq=faq,
                score=sum(weight * document.get(term, 0.0) for term, weight in query_vector.items()),
            )
            for faq, document in zip(self.faqs, self.vectors)
        ]
        results.sort(key=lambda result: (-result.score, result.faq.id))
        return results[:top_k]


def answer_question(
    query: str,
    retriever: FAQRetriever,
    *,
    top_k: int = 3,
    min_score: float = 0.05,
) -> str:
    """Retrieve a relevant FAQ answer and append a stable citation."""
    if not query.strip():
        return "Please enter a question so I can search the support FAQ."
    results = retriever.search(query, top_k=top_k)
    if not results or results[0].score < min_score:
        return FALLBACK_ANSWER
    faq = results[0].faq
    return f"{faq.answer}\n\nSource: [{faq.id}] {faq.category} FAQ"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the sample customer-support FAQ.")
    parser.add_argument("question", nargs="*", help="Question to ask")
    parser.add_argument("--faq", type=Path, default=DEFAULT_FAQ_PATH, help="Path to FAQ JSON")
    args = parser.parse_args()
    question = " ".join(args.question).strip()
    if not question:
        question = input("Ask a support question: " ).strip()
    retriever = FAQRetriever(load_faqs(args.faq))
    print(answer_question(question, retriever))


if __name__ == "__main__":
    main()
