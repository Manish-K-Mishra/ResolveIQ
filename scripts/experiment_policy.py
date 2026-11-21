"""Write an EXPERIMENTAL routing policy for a measured comparison; never the approved policy.

    python scripts/experiment_policy.py --retriever embedding --chunking section \
        --confidence 0.70 --retrieval 0.35 \
        --out evaluation/results/generation_comparison/policy_embedding_section.json

The approved policy (config/routing_policy.json, RP-1.0) supplies the restricted intents and the
intent-to-document approvals. The retrieval configuration and the threshold pair must be one that
`scripts/derive_routing_policy.py --search-retrievers` already measured on the development set
(evaluation/results/retrieval_comparison/policy_search.json), so the experiment uses only
dev-derived figures. If the embedding model is missing, the runtime falls back to RP-1.0's
lexical/article configuration and thresholds, and says so in every record.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPROVED = ROOT / "config/routing_policy.json"
SEARCH = ROOT / "evaluation/results/retrieval_comparison/policy_search.json"


def build(approved: dict, search: dict, retriever: str, chunking: str, conf: float, ret: float) -> dict:
    cfg = next((c for c in search["configurations"]
                if c["retriever"] == retriever and c["chunking"] == chunking), None)
    if cfg is None:
        raise SystemExit(f"{retriever}/{chunking} is not in {SEARCH.name}")
    cand = next((c for c in cfg["candidates_considered"] if c["classification_confidence_min"] == conf
                 and c["retrieval_score_min"] == ret), None)
    if cand is None:
        raise SystemExit(f"thresholds {conf}/{ret} were not measured for {retriever}/{chunking}")
    keep = ("restricted_intents", "intent_documents", "target_auto_precision", "confidence_threshold_validated_for")
    return {
        "policy_version": f"{approved['policy_version']}-experiment-{retriever}-{chunking}",
        "purpose": "experiment only (Slice 7 item 9); NOT the approved routing policy",
        "derived_from": f"{approved['policy_version']} approvals plus the dev out-of-fold search in {SEARCH.name}",
        **{k: approved[k] for k in keep},
        "retrieval": {k: cfg[k] for k in ("retriever", "name", "chunking", "min_score", "passages")},
        "classification_confidence_min": conf,
        "retrieval_score_min": ret,
        "dev_out_of_fold": cand,
        "retrieval_fallback": {**approved["retrieval"],
                               "classification_confidence_min": approved["classification_confidence_min"],
                               "retrieval_score_min": approved["retrieval_score_min"]},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--retriever", required=True)
    ap.add_argument("--chunking", required=True)
    ap.add_argument("--confidence", type=float, required=True)
    ap.add_argument("--retrieval", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if Path(a.out).resolve() == APPROVED.resolve():
        raise SystemExit("refusing to overwrite the approved routing policy")
    policy = build(json.loads(APPROVED.read_text()), json.loads(SEARCH.read_text()),
                   a.retriever, a.chunking, a.confidence, a.retrieval)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
    print(f"written {a.out}: {policy['policy_version']} dev out-of-fold {policy['dev_out_of_fold']}")


if __name__ == "__main__":
    main()
