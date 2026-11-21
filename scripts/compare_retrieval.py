"""Measure retrieval quality for every chunking x retriever configuration (Setup Guide §04).

    python scripts/compare_retrieval.py      # writes evaluation/results/retrieval_comparison/

Ground truth is labels.expected_doc_ids (answerable tickets) and labels.answerable_from_docs
(unanswerable tickets). Labels are read here for measurement only; retrieval never sees them.
Development tickets choose the configuration; validation is reported separately as a check.

Measures (document level: a passage counts as a hit for the article it comes from)
  hit@1          top passage's article is one of the expected articles
  hit@3          an expected article is among the first three distinct articles returned
  empty_answerable    answerable tickets for which retrieval returned nothing (lost answers)
  empty_unanswerable  unanswerable tickets for which retrieval returned nothing (the requirement:
                      "returns nothing rather than something irrelevant")
Each is reported at the retriever's default floor, plus a floor sweep, plus by language_fluency.
The output is deterministic for a given corpus, ticket files and model files.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from cloudserve.ingest import load_json, normalize
from cloudserve.retrieve import CHUNKINGS, KnowledgeBase, OnnxSentenceEmbedder

ROOT = Path(__file__).resolve().parents[1]
SWEEP = {"lexical": [0.26, 0.30, 0.35, 0.40, 0.45, 0.50],
         "embedding": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]}


def ranked_docs(hits: list[dict], floor: float) -> list[str]:
    out = []
    for h in hits:
        if h["score"] >= floor and h["doc_id"] not in out:
            out.append(h["doc_id"])
    return out


def frac(n, d):
    return round(n / d, 4) if d else None


def measure(rows: list[tuple[dict, list[dict]]], floor: float) -> dict:
    ans = [(t, ranked_docs(h, floor)) for t, h in rows if t["labels"].get("answerable_from_docs") and t["labels"].get("expected_doc_ids")]
    una = [(t, ranked_docs(h, floor)) for t, h in rows if t["labels"].get("answerable_from_docs") is False]
    exp = lambda t: set(t["labels"]["expected_doc_ids"])   # noqa: E731
    return {"answerable": len(ans), "unanswerable": len(una),
            "hit@1": frac(sum(bool(d) and d[0] in exp(t) for t, d in ans), len(ans)),
            "hit@3": frac(sum(bool(set(d[:3]) & exp(t)) for t, d in ans), len(ans)),
            "empty_answerable": frac(sum(not d for _, d in ans), len(ans)),
            "empty_unanswerable": frac(sum(not d for _, d in una), len(una))}


def run(tickets: list[dict], kb: KnowledgeBase, floor: float) -> dict:
    rows = [(t, kb.search(normalize(t).text, limit=50)) for t in tickets]
    by_fluency = {}
    for seg in sorted({str(t.get("language_fluency")) for t in tickets}):
        by_fluency[seg] = measure([r for r in rows if str(r[0].get("language_fluency")) == seg], floor)
    return {"default_floor": floor, **measure(rows, floor),
            "by_language_fluency": by_fluency,
            "floor_sweep": {f"{f:.2f}": measure(rows, f) for f in SWEEP[kb.retriever]}}


def table(results: dict, split: str) -> list[str]:
    lines = [f"### {split}", "",
             "| Chunking | Retriever | Passages | Floor | hit@1 | hit@3 | Empty (answerable) | Empty (unanswerable) "
             "| hit@3 fluent | hit@3 non-fluent |", "|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in results.items():
        m = r[split]
        f = m["by_language_fluency"]
        lines.append(f"| {r['chunking']} | {r['retriever']} | {r['passages']} | {m['default_floor']} | {m['hit@1']} | "
                     f"{m['hit@3']} | {m['empty_answerable']} | {m['empty_unanswerable']} | "
                     f"{f.get('fluent', {}).get('hit@3')} | {f.get('non_fluent', {}).get('hit@3')} |")
    return lines + [""]


def sweep_table(results: dict, split: str) -> list[str]:
    lines = [f"### Floor sweep ({split})", "",
             "| Chunking | Retriever | Floor | hit@1 | hit@3 | Empty (answerable) | Empty (unanswerable) |",
             "|---|---|---|---|---|---|---|"]
    for r in results.values():
        for f, m in r[split]["floor_sweep"].items():
            lines.append(f"| {r['chunking']} | {r['retriever']} | {f} | {m['hit@1']} | {m['hit@3']} | "
                         f"{m['empty_answerable']} | {m['empty_unanswerable']} |")
    return lines + [""]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare chunking and retrievers on labelled tickets")
    ap.add_argument("--docs", default=str(ROOT / "data/documentation.json"))
    ap.add_argument("--dev", default=str(ROOT / "data/development_tickets.json"))
    ap.add_argument("--validation", default=str(ROOT / "data/validation_tickets.json"))
    ap.add_argument("--out", default=str(ROOT / "evaluation/results/retrieval_comparison"))
    a = ap.parse_args(argv)
    docs, dev, val = load_json(a.docs), load_json(a.dev), load_json(a.validation)
    embedder = OnnxSentenceEmbedder()     # fails loudly if the model was not fetched
    results = {}
    for chunking in CHUNKINGS:
        for retriever in ("lexical", "embedding"):
            kb = KnowledgeBase(docs, chunking=chunking, retriever=retriever,
                               embedder=embedder if retriever == "embedding" else None)
            floor = kb.min_score
            probe = KnowledgeBase(docs, chunking=chunking, retriever=retriever, min_score=0.0,
                                  embedder=embedder if retriever == "embedding" else None)
            results[f"{chunking}/{retriever}"] = {
                "chunking": chunking, "retriever": retriever, "name": kb.name, "passages": len(kb.passages),
                "development": run(dev, probe, floor), "validation": run(val, probe, floor)}
            print(f"{chunking:8s} {retriever:9s} dev hit@1={results[f'{chunking}/{retriever}']['development']['hit@1']} "
                  f"hit@3={results[f'{chunking}/{retriever}']['development']['hit@3']} "
                  f"empty_unanswerable={results[f'{chunking}/{retriever}']['development']['empty_unanswerable']}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"corpus": a.docs, "development": a.dev, "validation": a.validation,
               "embedding_model": embedder.name, "measures": __doc__.split("Measures", 1)[1].split("Each is")[0].strip(),
               "results": results}
    (out / "retrieval_comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md = ["# Retrieval comparison", "",
          f"Corpus `data/documentation.json`; embedding model `{embedder.name}`. Document-level hits against "
          "`labels.expected_doc_ids`; empty rates at each retriever's default floor. "
          "Produced by `python scripts/compare_retrieval.py`.", ""]
    n_val = {k: v["validation"]["by_language_fluency"].get("non_fluent", {}).get("answerable") for k, v in results.items()}
    md += table(results, "development") + table(results, "validation")
    md += [f"Validation has only {max(n for n in n_val.values() if n is not None)} answerable non-fluent tickets, "
           "so its non-fluent column is not a reliable estimate; use the development figures.", ""]
    md += sweep_table(results, "development")
    (out / "retrieval_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
