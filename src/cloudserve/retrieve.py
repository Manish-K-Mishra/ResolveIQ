"""Retrieval (FR-03): lexical search over the approved CloudServe documentation."""
from __future__ import annotations
from pathlib import Path
from .ingest import load_json, tokens


class KnowledgeBase:
    """Lexical retrieval over whole articles.

    Chunking decision: each of the 29 articles is one passage. Articles are short and follow a
    fixed Symptoms/Causes/Resolution/Notes structure; splitting inside a resolution sequence
    yields passages that retrieve well but read as incomplete (Dataset Guide §2). The citation
    unit is therefore the article (doc_id); the quoted span inside it is verified verbatim.
    """
    MIN_SCORE = 0.26        # share of query terms found in the article
    MIN_OVERLAP = 2         # at least two shared terms, so single-word matches return nothing

    def __init__(self, docs: list[dict]):
        self.docs = []
        for d in docs:
            if not d.get("doc_id") or not d.get("content"):
                continue
            target = " ".join([str(d.get("title", "")), str(d.get("category", "")), str(d["content"])])
            self.docs.append((d, tokens(target)))
        self.by_id = {d["doc_id"]: d for d, _ in self.docs}

    @classmethod
    def from_path(cls, path: str | Path):
        return cls(load_json(path))

    def search(self, question: str, limit: int = 3) -> list[dict]:
        q = tokens(question)
        if not q:
            return []
        ranked = []
        for d, dt in self.docs:
            overlap = q & dt
            score = len(overlap) / len(q)
            if score >= self.MIN_SCORE and len(overlap) >= self.MIN_OVERLAP:
                ranked.append({"doc_id": d["doc_id"], "title": d.get("title", ""),
                               "score": round(score, 4), "passage": d["content"]})
        return sorted(ranked, key=lambda x: (-x["score"], x["doc_id"]))[:limit]
