"""Derive config/routing_policy.json from DEVELOPMENT tickets only (never validation).

Method
1. Restricted intents  = intents whose dev tickets carry labels.must_not_auto_respond = true.
2. Intent documents    = documents listed in labels.expected_doc_ids for that intent in dev.
3. Thresholds: the naive Bayes classifier is evaluated out-of-fold with 5-fold cross-validation
   GROUPED BY TICKET BODY (dev has 216 unique bodies across 500 tickets; plain CV would let the
   classifier memorise duplicates and overstate confidence). Each candidate pair
   (classification_confidence_min, retrieval_score_min) is run through the real routing function.
   Objective: maximise automatic-answer coverage subject to
     - zero must_not_auto_respond tickets routed to an automatic answer, and
     - auto-answer precision (share of auto decisions whose expected_route is auto_respond)
       >= TARGET_AUTO_PRECISION.
   The target is an owner-adjustable risk appetite, recorded in the policy file.
4. Retrieval: the policy uses lexical search over whole articles (owner-approved RP-1.0),
   recorded under `retrieval`. With --search-retrievers, step 3 runs for every chunking x
   retriever configuration (lexical and all-MiniLM-L6-v2 embedding; article, section, window)
   and the result is written to evaluation/results/retrieval_comparison/policy_search.json as
   evidence; the policy file is not touched. That search needs the embedding model
   (python scripts/fetch_models.py).

Run:  python scripts/derive_routing_policy.py [--dev data/development_tickets.json]
      python scripts/derive_routing_policy.py --search-retrievers
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from cloudserve.classify import NaiveBayesIntentClassifier
from cloudserve.ingest import load_json, normalize
from cloudserve.retrieve import CHUNKINGS, RETRIEVERS, KnowledgeBase, OnnxSentenceEmbedder
from cloudserve.route import route

POLICY_VERSION = "RP-1.0"
# The approved policy's retrieval configuration (owner-approved RP-1.0; docs/architecture.md §11).
POLICY_RETRIEVAL = {"retriever": "lexical", "chunking": "article"}
# --search-retrievers: every configuration measured by scripts/compare_retrieval.py (evidence only).
SEARCH_CANDIDATES = [{"retriever": r, "chunking": c} for r in RETRIEVERS for c in CHUNKINGS]
SEARCH_OUT = ROOT / "evaluation/results/retrieval_comparison/policy_search.json"
TARGET_AUTO_PRECISION = 0.80
CONF_GRID = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
RET_GRID = {"lexical": [0.26, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
            "embedding": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]}


def calibration(oof, bands=5):
    """Out-of-fold stated confidence vs observed intent accuracy (Evaluation Framework §3)."""
    rows = []
    for i in range(bands):
        lo, hi = i / bands, (i + 1) / bands
        g = [(cl["confidence"], cl["intent"] == t["labels"]["intent"]) for t, _, cl in oof
             if lo <= cl["confidence"] < hi or (i == bands - 1 and cl["confidence"] == 1.0)]
        if g:
            stated = sum(c for c, _ in g) / len(g)
            observed = sum(ok for _, ok in g) / len(g)
            rows.append({"band": f"{lo:.1f}-{hi:.1f}", "n": len(g), "stated": round(stated, 4),
                         "observed": round(observed, 4), "gap_points": round(100 * (stated - observed), 1)})
    return rows


def search_grid(oof, matches, base, ret_grid):
    """Every (confidence floor, retrieval floor) pair run through the real routing function."""
    candidates = []
    for c in CONF_GRID:
        for r in ret_grid:
            pol = {**base, "classification_confidence_min": c, "retrieval_score_min": r}
            auto = [t for (t, tk, cl), m in zip(oof, matches) if route(tk, cl, m, pol)["route"] == "answer"]
            correct = sum(t["labels"]["expected_route"] == "auto_respond" for t in auto)
            violations = sum(t["labels"]["must_not_auto_respond"] for t in auto)
            candidates.append({"classification_confidence_min": c, "retrieval_score_min": r,
                               "auto": len(auto), "auto_precision": round(correct / len(auto), 4) if auto else None,
                               "must_not_auto_respond_violations": violations})
    ok = [x for x in candidates if x["auto"] and x["must_not_auto_respond_violations"] == 0
          and x["auto_precision"] >= TARGET_AUTO_PRECISION]
    if not ok:
        return None, candidates
    # Safety tie-break: among candidates within 1% of the best coverage, prefer the higher
    # confidence floor (a margin against overconfident posteriors), then higher precision.
    top = max(x["auto"] for x in ok)
    near = [x for x in ok if x["auto"] >= 0.99 * top]
    best = max(near, key=lambda x: (x["classification_confidence_min"], x["auto_precision"], x["auto"]))
    return best, candidates


def out_of_fold(best, n, acc):
    return {"tickets": n, "intent_accuracy": round(acc, 4), "auto_answers": best["auto"],
            "auto_precision": best["auto_precision"],
            "must_not_auto_respond_violations": best["must_not_auto_respond_violations"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=str(ROOT / "data/development_tickets.json"))
    ap.add_argument("--docs", default=str(ROOT / "data/documentation.json"))
    ap.add_argument("--out", default=str(ROOT / "config/routing_policy.json"))
    ap.add_argument("--search-retrievers", action="store_true",
                    help="derive thresholds for every chunking x retriever configuration and write the "
                         "comparison to --search-out as evidence (does NOT write the policy file)")
    ap.add_argument("--search-out", default=str(SEARCH_OUT))
    a = ap.parse_args()
    dev = load_json(a.dev)
    docs = load_json(a.docs)

    restricted = sorted({t["labels"]["intent"] for t in dev if t["labels"]["must_not_auto_respond"]})
    intent_docs = defaultdict(set)
    for t in dev:
        intent_docs[t["labels"]["intent"]].update(t["labels"]["expected_doc_ids"])

    bodies = sorted({t["body"] for t in dev})
    random.Random(0).shuffle(bodies)
    fold_of = {b: i % 5 for i, b in enumerate(bodies)}
    oof = []
    for k in range(5):
        clf = NaiveBayesIntentClassifier([(f"{t['subject']} {t['body']}", t["labels"]["intent"])
                                          for t in dev if fold_of[t["body"]] != k])
        for t in dev:
            if fold_of[t["body"]] == k:
                ticket = normalize(t)
                oof.append((t, ticket, clf.classify(ticket)))
    oof_acc = sum(cl["intent"] == t["labels"]["intent"] for t, _, cl in oof) / len(oof)
    base = {"policy_version": "candidate", "restricted_intents": restricted,
            "intent_documents": {k: sorted(v) for k, v in intent_docs.items()}}

    # Each retrieval configuration gets its own thresholds, derived with its own scores and
    # exactly the passages and retriever the runtime builds (cloudserve.runtime).
    configs = SEARCH_CANDIDATES if a.search_retrievers else [POLICY_RETRIEVAL]
    embedder = OnnxSentenceEmbedder() if any(c["retriever"] == "embedding" for c in configs) else None
    evaluated = []
    for cfg in configs:
        kb = KnowledgeBase(docs, chunking=cfg["chunking"], retriever=cfg["retriever"],
                           embedder=embedder if cfg["retriever"] == "embedding" else None)
        matches = [kb.search(tk.text) for _, tk, _ in oof]
        best, candidates = search_grid(oof, matches, base, RET_GRID[cfg["retriever"]])
        top_precision = max((x for x in candidates if x["auto"]), key=lambda x: (x["auto_precision"], x["auto"]))
        evaluated.append({**kb.describe(), "meets_objective": best is not None, "best": best,
                          "highest_precision_candidate": top_precision, "candidates_considered": candidates})
    if a.search_retrievers:
        out = Path(a.search_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "purpose": "evidence only: thresholds derived per retrieval configuration with the RP-1.0 objective; "
                       "the routing policy is not changed by this search (docs/architecture.md §11)",
            "objective": f"0 must_not_auto_respond violations and auto precision >= {TARGET_AUTO_PRECISION}, maximise coverage",
            "dev_out_of_fold_intent_accuracy": round(oof_acc, 4), "configurations": evaluated}, indent=2) + "\n",
            encoding="utf-8")
        for e in evaluated:
            print(f"{e['chunking']:8s} {e['retriever']:9s} meets={e['meets_objective']} best={e['best']}")
        print(f"written {out}")
        return
    kb_desc, best, candidates = evaluated[0], evaluated[0]["best"], evaluated[0]["candidates_considered"]
    if best is None:
        raise SystemExit("No threshold pair meets the objective; review TARGET_AUTO_PRECISION")
    policy = {
        "policy_version": POLICY_VERSION,
        "derived_on": date.today().isoformat(),
        "derived_from": "data/development_tickets.json only; 5-fold out-of-fold predictions grouped by ticket body",
        "objective": (f"maximise auto-answer coverage s.t. 0 must_not_auto_respond violations and auto precision >= "
                      f"{TARGET_AUTO_PRECISION}; among candidates within 1% of best coverage prefer the higher confidence floor"),
        "target_auto_precision": TARGET_AUTO_PRECISION,
        "restricted_intents": restricted,
        "intent_documents": {k: sorted(v) for k, v in sorted(intent_docs.items())},
        "retrieval": {k: kb_desc[k] for k in ("retriever", "name", "chunking", "min_score", "passages")},
        "classification_confidence_min": best["classification_confidence_min"],
        # The floor is derived from naive Bayes posteriors; an LLM's self-reported score is on a
        # different scale, so runs with --classifier llm are recorded as using an unvalidated floor.
        "confidence_threshold_validated_for": "naive_bayes_intent",
        "dev_out_of_fold_calibration": calibration(oof),
        "retrieval_score_min": best["retrieval_score_min"],
        "dev_out_of_fold": out_of_fold(best, len(oof), oof_acc),
        "candidates_considered": candidates,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"retrieval": policy["retrieval"],
                      **{k: policy[k] for k in ("classification_confidence_min", "retrieval_score_min", "dev_out_of_fold")}},
                     indent=2))


if __name__ == "__main__":
    main()
