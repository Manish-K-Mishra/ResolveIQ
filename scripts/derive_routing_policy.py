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

Run:  python scripts/derive_routing_policy.py [--dev data/development_tickets.json]
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
from cloudserve.retrieve import KnowledgeBase
from cloudserve.route import route

TARGET_AUTO_PRECISION = 0.80
CONF_GRID = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
RET_GRID = [0.26, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def calibration(oof, bands=5):
    """Out-of-fold stated confidence vs observed intent accuracy (Evaluation Framework §3)."""
    rows = []
    for i in range(bands):
        lo, hi = i / bands, (i + 1) / bands
        g = [(cl["confidence"], cl["intent"] == t["labels"]["intent"]) for t, _, cl, _ in oof
             if lo <= cl["confidence"] < hi or (i == bands - 1 and cl["confidence"] == 1.0)]
        if g:
            stated = sum(c for c, _ in g) / len(g)
            observed = sum(ok for _, ok in g) / len(g)
            rows.append({"band": f"{lo:.1f}-{hi:.1f}", "n": len(g), "stated": round(stated, 4),
                         "observed": round(observed, 4), "gap_points": round(100 * (stated - observed), 1)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=str(ROOT / "data/development_tickets.json"))
    ap.add_argument("--docs", default=str(ROOT / "data/documentation.json"))
    ap.add_argument("--out", default=str(ROOT / "config/routing_policy.json"))
    a = ap.parse_args()
    dev = load_json(a.dev)
    kb = KnowledgeBase.from_path(a.docs)

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
                oof.append((t, ticket, clf.classify(ticket), kb.search(ticket.text)))

    candidates = []
    base = {"policy_version": "candidate", "restricted_intents": restricted,
            "intent_documents": {k: sorted(v) for k, v in intent_docs.items()}}
    for c in CONF_GRID:
        for r in RET_GRID:
            pol = {**base, "classification_confidence_min": c, "retrieval_score_min": r}
            auto = [t for t, tk, cl, m in oof if route(tk, cl, m, pol)["route"] == "answer"]
            correct = sum(t["labels"]["expected_route"] == "auto_respond" for t in auto)
            violations = sum(t["labels"]["must_not_auto_respond"] for t in auto)
            candidates.append({"classification_confidence_min": c, "retrieval_score_min": r,
                               "auto": len(auto), "auto_precision": round(correct / len(auto), 4) if auto else None,
                               "must_not_auto_respond_violations": violations})
    ok = [x for x in candidates if x["auto"] and x["must_not_auto_respond_violations"] == 0
          and x["auto_precision"] >= TARGET_AUTO_PRECISION]
    if not ok:
        raise SystemExit("No threshold pair meets the objective; review TARGET_AUTO_PRECISION")
    # Safety tie-break: among candidates within 1% of the best coverage, prefer the higher
    # confidence floor (a margin against overconfident posteriors), then higher precision.
    top = max(x["auto"] for x in ok)
    near = [x for x in ok if x["auto"] >= 0.99 * top]
    best = max(near, key=lambda x: (x["classification_confidence_min"], x["auto_precision"], x["auto"]))
    oof_acc = sum(cl["intent"] == t["labels"]["intent"] for t, _, cl, _ in oof) / len(oof)
    policy = {
        "policy_version": "RP-1.0",
        "derived_on": date.today().isoformat(),
        "derived_from": "data/development_tickets.json only; 5-fold out-of-fold predictions grouped by ticket body",
        "objective": (f"maximise auto-answer coverage s.t. 0 must_not_auto_respond violations and auto precision >= "
                      f"{TARGET_AUTO_PRECISION}; among candidates within 1% of best coverage prefer the higher confidence floor"),
        "target_auto_precision": TARGET_AUTO_PRECISION,
        "restricted_intents": restricted,
        "intent_documents": {k: sorted(v) for k, v in sorted(intent_docs.items())},
        "classification_confidence_min": best["classification_confidence_min"],
        # The floor is derived from naive Bayes posteriors; an LLM's self-reported score is on a
        # different scale, so runs with --classifier llm are recorded as using an unvalidated floor.
        "confidence_threshold_validated_for": "naive_bayes_intent",
        "dev_out_of_fold_calibration": calibration(oof),
        "retrieval_score_min": best["retrieval_score_min"],
        "dev_out_of_fold": {"tickets": len(oof), "intent_accuracy": round(oof_acc, 4),
                            "auto_answers": best["auto"], "auto_precision": best["auto_precision"],
                            "must_not_auto_respond_violations": best["must_not_auto_respond_violations"]},
        "candidates_considered": candidates,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: policy[k] for k in ("classification_confidence_min", "retrieval_score_min", "dev_out_of_fold")}, indent=2))


if __name__ == "__main__":
    main()
