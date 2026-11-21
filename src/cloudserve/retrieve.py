"""Retrieval (FR-03): ranked, traceable passages from the approved CloudServe documentation.

Build Specification ("Retrieve"): search the supplied corpus and return ranked passages with
scores; return identifiers that resolve to the real corpus; apply a relevance threshold and
return nothing rather than something irrelevant.

Passages. Every passage has a stable id `DOC-XXX#<slug>` and character offsets into the
original article `content`, so `content[start:end] == text` exactly and a citation can be
followed back to the source. Three chunking configurations are supported and were measured
(scripts/compare_retrieval.py, docs/architecture.md §11):
  - article : one passage per article                      (ids DOC-XXX#article)
  - section : one passage per "## " section of the article (ids DOC-XXX#symptoms, #resolution, ...)
  - window  : ~800-character windows with 120 overlap      (ids DOC-XXX#w1, #w2, ...)

Retrievers. `lexical` scores the share of query terms found in the passage (no dependencies).
`embedding` is cosine similarity of all-MiniLM-L6-v2 sentence embeddings, run on CPU through
ONNX Runtime from files fetched by `python scripts/fetch_models.py`. The index is an in-memory
matrix: the corpus is about a hundred passages, so a vector database adds nothing here.
The article title is prefixed to each passage for indexing only (it is not part of the passage
text, offsets or quotes), so a short section still says which article it belongs to.

Which retriever and chunking run is set by the routing policy (`retrieval` in
config/routing_policy.json) and built by cloudserve.runtime.build_knowledge_base, so search
always matches the configuration the thresholds were derived for.
"""
from __future__ import annotations
import re
from pathlib import Path
from . import ROOT
from .ingest import load_json, tokens

CHUNKINGS = ("article", "section", "window")
RETRIEVERS = ("lexical", "embedding")
WINDOW_CHARS, WINDOW_OVERLAP = 800, 120
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"   # pinned in scripts/fetch_models.py
DEFAULT_MODEL_DIR = ROOT / "storage" / "models" / MODEL_NAME
_HEADING = re.compile(r"(?m)^## +(.+?)\s*$")
RESOLUTION_SECTION = re.compile(r"(?i)(?:Resolution|Steps|Workaround|How to|Fix|Recommended actions)\b.*")


class EmbeddingModelUnavailable(RuntimeError):
    pass


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _trimmed(content: str, start: int, end: int) -> tuple[int, int]:
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    return start, end


def _spans(content: str, chunking: str) -> list[tuple[str, str, int, int]]:
    """(slug, section name, start, end) spans of `content` for one chunking configuration."""
    if chunking == "article":
        return [("article", "article", *_trimmed(content, 0, len(content)))]
    if chunking == "section":
        heads = list(_HEADING.finditer(content))
        out, seen = [], {}
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(content)
            slug = _slug(h.group(1))
            seen[slug] = seen.get(slug, 0) + 1
            if seen[slug] > 1:
                slug = f"{slug}-{seen[slug]}"
            out.append((slug, h.group(1), *_trimmed(content, h.start(), end)))
        # An article without headings is still searchable as a whole.
        return out or [("article", "article", *_trimmed(content, 0, len(content)))]
    if chunking == "window":
        out, start, n = [], 0, len(content)
        while start < n:
            end = min(start + WINDOW_CHARS, n)
            if end < n:   # end on a line or word break, never mid-word
                cut = max(content.rfind("\n", start + WINDOW_CHARS // 2, end), content.rfind(" ", start + WINDOW_CHARS // 2, end))
                end = cut if cut > start else end
            s, e = _trimmed(content, start, end)
            if e > s:
                out.append((f"w{len(out) + 1}", "window", s, e))
            if end >= n:
                break
            nxt = max(end - WINDOW_OVERLAP, start + 1)
            while nxt < end and not content[nxt - 1].isspace():   # start on a word boundary
                nxt += 1
            start = nxt
        return out
    raise ValueError(f"unknown chunking {chunking!r}; choose from {CHUNKINGS}")


def build_passages(docs: list[dict], chunking: str) -> list[dict]:
    passages = []
    for d in docs:
        if not d.get("doc_id") or not d.get("content"):
            continue
        content = str(d["content"])
        for slug, section, start, end in _spans(content, chunking):
            passages.append({"passage_id": f"{d['doc_id']}#{slug}", "doc_id": d["doc_id"],
                             "title": str(d.get("title", "")), "category": str(d.get("category", "")),
                             "section": section, "start": start, "end": end, "text": content[start:end]})
    return passages


class OnnxSentenceEmbedder:
    """all-MiniLM-L6-v2 on CPU: tokenise, run the ONNX encoder, mean-pool, L2-normalise.

    Texts are embedded one at a time (no padding), so a vector never depends on what else was
    in the batch; that keeps rankings identical from run to run.
    """
    MAX_TOKENS = 256    # the sentence-transformers max_seq_length for this model

    def __init__(self, model_dir: str | Path | None = None):
        model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        onnx_file, tok_file = model_dir / "model.onnx", model_dir / "tokenizer.json"
        if not (onnx_file.exists() and tok_file.exists()):
            raise EmbeddingModelUnavailable(f"model files not found in {model_dir} (run: python scripts/fetch_models.py)")
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise EmbeddingModelUnavailable(f"embedding packages missing: {exc.name}") from exc
        self._np = np
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1      # deterministic and cheap at this scale
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(tok_file))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=self.MAX_TOKENS)
        self.name = f"{MODEL_NAME}@{MODEL_REVISION[:7]}"

    def embed(self, texts: list[str]):
        np = self._np
        rows = []
        for text in texts:
            enc = self.tokenizer.encode(text or " ")
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            hidden = self.session.run(None, {"input_ids": ids, "attention_mask": mask,
                                             "token_type_ids": np.array([enc.type_ids], dtype=np.int64)})[0]
            m = mask[..., None].astype(np.float32)
            v = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            rows.append(v[0] / max(float(np.linalg.norm(v[0])), 1e-12))
        return np.vstack(rows).astype(np.float32) if rows else np.zeros((0, 384), dtype=np.float32)


class KnowledgeBase:
    """Ranked passage search with a relevance floor. Same input, same ranked passages.

    `search` returns at most `limit` passages, best first, each with passage_id, doc_id, title,
    section, start/end offsets, the exact passage text (`passage`) and a `score` in [0, 1].
    Passages below the retriever's floor are never returned, so an irrelevant query gets [].
    """
    LEXICAL_MIN_SCORE = 0.26    # share of query terms found in the passage
    LEXICAL_MIN_OVERLAP = 2     # at least two shared terms, so single-word matches return nothing
    EMBEDDING_MIN_SCORE = 0.35  # cosine floor; chosen from the dev measurement (docs/architecture.md §11)

    def __init__(self, docs: list[dict], chunking: str = "article", retriever: str = "lexical",
                 embedder=None, min_score: float | None = None):
        if retriever not in RETRIEVERS:
            raise ValueError(f"unknown retriever {retriever!r}; choose from {RETRIEVERS}")
        self.chunking, self.retriever = chunking, retriever
        self.passages = build_passages(docs, chunking)
        self.by_id = {d["doc_id"]: d for d in docs if d.get("doc_id") and d.get("content")}
        self.passage_by_id = {p["passage_id"]: p for p in self.passages}
        # Answer units: each article's complete resolution section, whatever the search chunking,
        # so an answer never quotes a window that cuts a numbered sequence in half.
        self.resolution_by_doc = {}
        for p in build_passages(docs, "section"):
            if RESOLUTION_SECTION.fullmatch(p["section"]) and p["doc_id"] not in self.resolution_by_doc:
                self.resolution_by_doc[p["doc_id"]] = p
                self.passage_by_id.setdefault(p["passage_id"], p)
        self.docs = [(self.by_id[p["doc_id"]], p) for p in self.passages]   # kept for introspection
        self.min_score = min_score if min_score is not None else (
            self.EMBEDDING_MIN_SCORE if retriever == "embedding" else self.LEXICAL_MIN_SCORE)
        indexed = [f"{p['title']}. {p['text']}" for p in self.passages]
        if retriever == "embedding":
            if embedder is None:
                raise EmbeddingModelUnavailable("embedding retriever needs an embedder")
            self.embedder = embedder
            self.matrix = embedder.embed(indexed) if indexed else None
            self.name = f"embedding:{embedder.name}"
        else:
            self.embedder, self.matrix = None, None
            self._tokens = [tokens(" ".join([p["title"], p["category"], p["text"]])) for p in self.passages]
            self.name = "lexical:term-overlap"

    @classmethod
    def from_path(cls, path: str | Path, **kwargs):
        return cls(load_json(path), **kwargs)

    def describe(self) -> dict:
        return {"retriever": self.retriever, "name": self.name, "chunking": self.chunking,
                "min_score": self.min_score, "passages": len(self.passages)}

    def resolve(self, passage_id: str) -> str | None:
        """Exact corpus text for a passage id (None if the id does not exist)."""
        p = self.passage_by_id.get(passage_id)
        return None if p is None else self.by_id[p["doc_id"]]["content"][p["start"]:p["end"]]

    def answer_passages(self, evidence: list[dict]) -> list[dict]:
        """Resolution-section passages for the approved evidence, best-matching article first.

        `evidence` is the routing policy's approved, above-threshold passages (best first). Each
        article contributes its complete resolution section once, carrying the score of that
        article's best evidence passage. Articles without a resolution section contribute the
        evidence passage itself (the selector then finds no steps and the answer is blocked).
        """
        out, seen = [], set()
        for m in evidence:
            if m["doc_id"] in seen:
                continue
            seen.add(m["doc_id"])
            p = self.resolution_by_doc.get(m["doc_id"])
            if p is None:
                out.append(dict(m, matched_passage_id=m["passage_id"]))
                continue
            out.append({"passage_id": p["passage_id"], "doc_id": p["doc_id"], "title": p["title"],
                        "section": p["section"], "start": p["start"], "end": p["end"],
                        "score": m["score"], "passage": p["text"], "matched_passage_id": m["passage_id"]})
        return out

    def _scores(self, question: str) -> list[float]:
        if self.retriever == "embedding":
            q = self.embedder.embed([question])[0]
            return [float(s) for s in self.matrix @ q]
        q = tokens(question)
        out = []
        for pt in self._tokens:
            overlap = q & pt
            ok = len(overlap) >= self.LEXICAL_MIN_OVERLAP
            out.append(len(overlap) / len(q) if ok else 0.0)
        return out

    def search(self, question: str, limit: int = 5) -> list[dict]:
        if not self.passages or not tokens(question or ""):
            return []
        ranked = []
        for p, score in zip(self.passages, self._scores(question)):
            score = round(min(max(score, 0.0), 1.0), 4)
            if score >= self.min_score and score > 0:
                ranked.append({"passage_id": p["passage_id"], "doc_id": p["doc_id"], "title": p["title"],
                               "section": p["section"], "start": p["start"], "end": p["end"],
                               "score": score, "passage": p["text"]})
        return sorted(ranked, key=lambda x: (-x["score"], x["passage_id"]))[:limit]
