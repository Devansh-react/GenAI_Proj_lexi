"""Deterministic local lexical search over the ledger's ingested chunks."""
from __future__ import annotations
from collections import Counter
import re


def _terms(text: str) -> list[str]:
    # Keep Unicode word characters so identifiers such as ₹45,000 and normal
    # legal names are tokenized consistently across the corpus and questions.
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class LexicalSearch:
    def search(self, question: str, chunks: list[dict], top_k: int) -> list[tuple[dict, float]]:
        query = Counter(_terms(question))
        scored: list[tuple[dict, float]] = []
        for chunk in chunks:
            # A legal clause is easier to find when the visible section/title
            # label participates in lexical matching (e.g. “Unit 4B” is in the
            # document title while “monthly rent” is in the Rent clause).
            searchable = " ".join(
                str(chunk.get(field, ""))
                for field in ("doc_title", "section_title", "text")
            )
            terms = Counter(_terms(searchable))
            overlap = sum(min(count, terms[word]) for word, count in query.items())
            # Recall-oriented lexical support: a short clause should not lose
            # merely because the question contains polite/auxiliary wording.
            score = overlap / max(1, min(len(query), len(terms)))
            if score:
                scored.append((chunk, float(score)))
        return sorted(scored, key=lambda item: (-item[1], item[0]["chunk_id"]))[:top_k]
