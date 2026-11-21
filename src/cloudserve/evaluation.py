"""Unattended evaluation harness (A9, A10).

    python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/<run>
    python -m cloudserve.evaluation ...   (same entry point)

Runs the SAME decision engine as the CLI over every ticket in an arbitrary input file, writes
decisions.json, a fresh decisions.sqlite audit log, metrics.json and metrics_summary.md.
Labels (if the file has them) are read only here, after all decisions are made, for measurement.
Unlabelled files (e.g. the hidden set without labels) still produce volume/governance metrics.
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from .guardrails import PII_PATTERN, mask_pii
from .ingest import load_tickets
from .logging_store import new_run_id
from .runtime import add_engine_arguments, build_engine, run_batch


def frac(n, d):
    return round(n / d, 4) if d else None


def pct_points(values):
    vals = [v for v in values if v is not None]
    return round(100 * (max(vals) - min(vals)), 1) if len(vals) > 1 else None


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    import math   # nearest-rank percentile
    return sorted_vals[max(0, math.ceil(q * len(sorted_vals)) - 1)]


def per_class(actual, predicted):
    out = {}
    for k in sorted(set(actual) | set(predicted)):
        tp = sum(a == k == p for a, p in zip(actual, predicted))
        fp = sum(a != k and p == k for a, p in zip(actual, predicted))
        fn = sum(a == k and p != k for a, p in zip(actual, predicted))
        out[k] = {"support": actual.count(k), "predicted": predicted.count(k),
                  "precision": frac(tp, tp + fp), "recall": frac(tp, tp + fn)}
    return out


def macro(report, key, only_supported=True):
    vals = [v[key] for v in report.values() if v[key] is not None and (v["support"] or not only_supported)]
    return round(statistics.mean(vals), 4) if vals else None


def confusion(actual, predicted):
    m = defaultdict(Counter)
    for a, p in zip(actual, predicted):
        m[a][p] += 1
    return {a: dict(sorted(c.items())) for a, c in sorted(m.items())}


def calibration_table(rows, bands=5):
    """Evaluation Framework §3: stated confidence vs observed accuracy per band."""
    out = []
    for i in range(bands):
        lo, hi = i / bands, (i + 1) / bands
        g = [(c, ok) for c, ok in rows if lo <= c < hi or (i == bands - 1 and c == 1.0)]
        if g:
            stated = statistics.mean(c for c, _ in g)
            observed = sum(ok for _, ok in g) / len(g)
            out.append({"band": f"{lo:.1f}-{hi:.1f}", "n": len(g), "stated": round(stated, 4),
                        "observed": round(observed, 4), "gap_points": round(100 * (stated - observed), 1)})
    return out


def cited_doc(r: dict) -> str:
    """Article of the first citation (citations are passage ids DOC-XXX#<slug>)."""
    return r["citations"][0].split("#", 1)[0]


def top_docs(r: dict, k: int = 3) -> list[str]:
    """First k distinct articles among the retrieved passages, best first."""
    out = []
    for s in r["sources_used"]:
        if s["doc_id"] not in out:
            out.append(s["doc_id"])
    return out[:k]


def citations_resolve(results: list[dict], kb) -> float | None:
    """Share of cited quotes found at their recorded span in the real corpus text."""
    spans = [sp for r in results if r["response"] for sp in r.get("citation_spans", [])]
    ok = 0
    for sp in spans:
        text = kb.resolve(sp["passage_id"])
        doc = kb.by_id.get(sp["doc_id"], {}).get("content", "")
        ok += bool(text) and sp["start"] is not None and doc[sp["start"]:sp["end"]] in text and sp["end"] > sp["start"]
    return frac(ok, len(spans))


GROUNDING_FAILURES = ("invalid_citation", "quote_not_supported_by_source", "unsupported_claim")
REVIEW_SAMPLE_SIZE = 50
REVIEW_COLUMNS = ["ticket_id", "ticket_text_masked", "response", "cited_passages_text",
                  "assessor_1_supported", "assessor_2_supported",
                  "assessor_1_quality_1to5", "assessor_2_quality_1to5", "notes"]


def generation_metrics(results: list[dict]) -> dict | None:
    """Generated-answer measures (answer_mode generated). Support is an AUTOMATED PROXY: the
    deterministic support check of guardrails.validate_generated, not a human judgement."""
    gen = [r for r in results if r.get("answer_mode") == "generated"]
    if not gen:
        return None
    answered = [r for r in gen if r["action_taken"] == "auto_respond"]
    per_cite = [ok for r in answered for e in (r.get("generation") or {}).get("sentences", [])
                for ok in e["citation_support"].values()]
    checked = [ok for r in gen if r.get("generation") for e in r["generation"]["sentences"]
               for ok in e["citation_support"].values()]
    return {
        "generator_called": sum(r.get("generation") is not None for r in gen),
        "answers_released_or_held": len(answered),
        "citations_per_answer": round(statistics.mean(len(r["citations"]) for r in answered), 3) if answered else None,
        "sentences_per_answer": round(statistics.mean(len(r["generation"]["sentences"]) for r in answered), 3) if answered else None,
        "citation_support_rate_automated_proxy": frac(sum(per_cite), len(per_cite)),
        "citation_support_rate_all_generated_automated_proxy": frac(sum(checked), len(checked)),
        "support_rate_label": "automated proxy (deterministic string checks), not human-reviewed citation accuracy",
        "blocked_by_grounding": sum(r["action_taken"] == "block" and bool(set(r["guardrail_failures"]) & set(GROUNDING_FAILURES)) for r in gen),
        "cannot_answer": sum("documentation_does_not_answer" in r["reason_codes"] for r in gen),
        "generation_failed": sum("generation_failed" in r["reason_codes"] for r in gen),
        "unsupported_claims_flagged": sum(len(r.get("unsupported_claims", [])) for r in gen),
    }


def write_human_review_sample(items: list, results: list[dict], kb, path) -> int:
    """First REVIEW_SAMPLE_SIZE answered (auto_respond, sent or held) decisions in input order, for
    two assessors (Evaluation Framework: hallucination needs >=50 human-reviewed responses).
    Assessor columns are left empty on purpose."""
    rows = []
    for t, r in zip(items, results):
        if r["action_taken"] != "auto_respond" or len(rows) >= REVIEW_SAMPLE_SIZE:
            continue
        t = t if isinstance(t, dict) else {}
        text = mask_pii(f"{t.get('subject') or ''}\n{t.get('body') or ''}".strip())
        cited = "\n---\n".join(f"[{pid}] {kb.resolve(pid) if kb is not None else ''}" for pid in r["citations"])
        rows.append({"ticket_id": r["ticket_id"], "ticket_text_masked": text, "response": r["response"],
                     "cited_passages_text": cited, **{c: "" for c in REVIEW_COLUMNS[4:]}})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def evaluate(items: list, results: list[dict], logged: int, wall_clock: float, run_info: dict, kb=None) -> dict:
    n = len(items)
    st = Counter(r["status"] for r in results)
    act = Counter(r["action_taken"] for r in results)
    sent = [r for r in results if r["answer_sent"]]
    lat = sorted(r["latency_seconds"] for r in results)
    sent_lat = sorted(r["latency_seconds"] for r in sent)
    outbound_pii = sum(bool(PII_PATTERN.search(r["response"])) for r in results if r["response"])
    guard_types = Counter(f for r in results for f in r["guardrail_failures"])
    for r in results:
        for code in r["reason_codes"]:
            if code in ("untrusted_instruction_in_ticket", "sensitive_request_manual_review"):
                guard_types[code] += 1

    report = {
        "run": {**run_info, "tickets_in_file": n, "wall_clock_seconds": round(wall_clock, 2)},
        "volume": {"processed": len(results), "decisions_logged": logged,
                   "auto_responded_sent": st.get("auto_responded", 0),
                   "auto_respond_decisions_held_by_kill_switch": st.get("held_for_review", 0),
                   "escalated": st.get("escalated", 0), "blocked_by_guardrails": st.get("blocked", 0)},
        "business": {
            "first_contact_resolution_pct": round(100 * len(sent) / n, 1) if n else None,
            "first_contact_resolution_if_kill_switch_released_pct": round(100 * act.get("auto_respond", 0) / n, 1) if n else None,
            "escalation_rate_pct": round(100 * (n - len(sent)) / n, 1) if n else None,
            "escalation_rate_if_kill_switch_released_pct": round(100 * (act.get("escalate", 0) + act.get("block", 0)) / n, 1) if n else None,
            "time_to_first_reply_automatic_seconds": {
                "n": len(sent_lat), "mean": round(statistics.mean(sent_lat), 3) if sent_lat else None,
                "median": round(statistics.median(sent_lat), 3) if sent_lat else None},
            "time_to_first_reply_escalated": "not measurable offline: depends on the human queue",
            "customer_satisfaction": "not measurable offline: requires a human-scored sample",
            "repeat_contacts": "not measurable offline: requires a 7-day follow-up window",
        },
        "technical": {"latency_seconds": {"mean": round(statistics.mean(lat), 4) if lat else None,
                                          "median": round(statistics.median(lat), 4) if lat else None,
                                          "p95": percentile(lat, 0.95)}},
        "governance": {
            "decisions_logged": logged, "tickets_processed": len(results),
            "decision_log_reconciles": logged == len(results) == n,
            "guardrail_activations_by_type": dict(sorted(guard_types.items())),
            "private_data_detections": guard_types.get("possible_private_data", 0),
            "private_data_in_outbound_responses": outbound_pii,
            "citations_resolving_to_corpus_text": citations_resolve(results, kb) if kb is not None else None,
        },
    }
    gen = generation_metrics(results)
    if gen is not None:
        report["generation"] = gen

    labelled = [(t, r) for t, r in zip(items, results) if isinstance(t, dict) and isinstance(t.get("labels"), dict)]
    report["labelled_tickets"] = len(labelled)
    if labelled:
        L = [t["labels"] for t, _ in labelled]
        R = [r for _, r in labelled]
        yi, pi = [l.get("intent", "unknown") for l in L], [r["prediction"]["value"] for r in R]
        yu, pu = [l.get("urgency", "unknown") for l in L], [str(r["prediction"]["urgency"]) for r in R]
        intent_rep = per_class(yi, pi)
        auto = [(l, r) for l, r in zip(L, R) if r["action_taken"] == "auto_respond"]
        with_docs = [(l, r) for l, r in zip(L, R) if l.get("expected_doc_ids")]
        unanswerable = [(l, r) for l, r in zip(L, R) if l.get("answerable_from_docs") is False]
        cited = [(l, r) for l, r in auto if r["citations"]]
        cited_with_docs = [(l, r) for l, r in cited if l.get("expected_doc_ids")]
        report["technical"].update({
            "intent_accuracy": frac(sum(a == p for a, p in zip(yi, pi)), len(yi)),
            "intent_macro_precision": macro(intent_rep, "precision"),
            "intent_macro_recall": macro(intent_rep, "recall"),
            "intent_precision_recall_by_class": intent_rep,
            "intent_confusion_matrix": confusion(yi, pi),
            "urgency_accuracy": frac(sum(a == p for a, p in zip(yu, pu)), len(yu)),
            "urgency_precision_recall_by_class": per_class(yu, pu),
            "retrieval_hit_rate_any_expected_doc": frac(sum(bool(set(l["expected_doc_ids"]) & set(top_docs(r))) for l, r in with_docs), len(with_docs)),
            "retrieval_hit_rate_top1": frac(sum(bool(r["sources_used"]) and r["sources_used"][0]["doc_id"] in l["expected_doc_ids"] for l, r in with_docs), len(with_docs)),
            "retrieval_eligible_tickets": len(with_docs),
            "retrieval_returned_nothing_for_unanswerable": frac(sum(not r["sources_used"] for _, r in unanswerable), len(unanswerable)),
            "citation_accuracy_vs_expected_docs": frac(sum(cited_doc(r) in l["expected_doc_ids"] for l, r in cited_with_docs), len(cited_with_docs)),
            "citations_on_tickets_labelled_unanswerable": sum(not l.get("expected_doc_ids") for l, _ in cited),
            "citation_article_in_retrieved_set": frac(sum(cited_doc(r) in {s["doc_id"] for s in r["sources_used"]} for _, r in cited), len(cited)),
        })
        exp_auto = [l.get("expected_route") == "auto_respond" for l in L]
        dec_auto = [r["action_taken"] == "auto_respond" for r in R]
        tp = sum(e and d for e, d in zip(exp_auto, dec_auto))
        report["routing"] = {
            "policy_version": R[0]["policy_version"],
            "auto_respond_decisions": sum(dec_auto),
            "auto_precision_vs_expected_route": frac(tp, sum(dec_auto)),
            "auto_recall_vs_expected_route": frac(tp, sum(exp_auto)),
            "routing_accuracy_vs_expected_route": frac(sum(e == d for e, d in zip(exp_auto, dec_auto)), len(L)),
            "confusion_expected_vs_decided": confusion(["auto_respond" if e else "escalate" for e in exp_auto],
                                                       [r["action_taken"] for r in R]),
            "must_not_auto_respond_tickets": sum(bool(l.get("must_not_auto_respond")) for l in L),
            "must_not_auto_respond_violations": sum(bool(l.get("must_not_auto_respond")) and d for l, d in zip(L, dec_auto)),
        }
        cal_rows = [(float(r["prediction"]["confidence"]), r["prediction"]["value"] == l.get("intent"))
                    for l, r in zip(L, R) if r["prediction"]["confidence_type"] not in ("model_failed", "not_classified")]
        table = calibration_table(cal_rows)
        report["governance"]["confidence_calibration"] = {
            "bands": table, "max_abs_gap_points": max((abs(b["gap_points"]) for b in table), default=None),
            "target": "within 5 points per band"}
        segments = {}
        for field in ("customer_tier", "language_fluency", "customer_region", "channel"):
            groups = defaultdict(list)
            for t, r in labelled:
                groups[str(t.get(field))].append((t["labels"], r))
            seg = {}
            for g, rows in sorted(groups.items()):
                a = [r["action_taken"] == "auto_respond" for _, r in rows]
                seg[g] = {"n": len(rows),
                          "auto_respond_rate": frac(sum(a), len(rows)),
                          "routing_accuracy": frac(sum((l.get("expected_route") == "auto_respond") == x for (l, _), x in zip(rows, a)), len(rows)),
                          "intent_accuracy": frac(sum(r["prediction"]["value"] == l.get("intent") for l, r in rows), len(rows))}
            segments[field] = {"groups": seg,
                               "variation_points": {k: pct_points([v[k] for v in seg.values()])
                                                    for k in ("auto_respond_rate", "routing_accuracy", "intent_accuracy")}}
        report["fairness_segments"] = segments
    ov = run_info.get("training_overlap", {})
    report["limitations"] = [
        (f"Training overlap: {ov.get('exact_body_matches', 'an unknown number')} of {ov.get('tickets', n)} ticket bodies "
         "in this input are exact copies of development tickets the offline classifier was trained on"
         + ("; intent accuracy on those measures recall of training data, not generalisation, so treat it with caution "
            "(docs/architecture.md §8)." if ov.get("exact_body_matches") != 0 else
            ", so these figures are not inflated by training-set recall.")),
        "Urgency and expected_route labels vary for identical ticket text in the supplied data, which caps achievable urgency and routing accuracy.",
        "Citation accuracy here is a label-based proxy (cited document among expected_doc_ids); it is not the human claim-by-claim review the Evaluation Framework requires, and hallucination rate is not measured.",
        ("Responses are generated sentences (PR-02); the citation support rate is an automated proxy from "
         "deterministic string checks, so hallucination still needs the two-assessor human review "
         "(human_review_sample.csv)." if gen is not None else
         "Responses are extractive (a verbatim documentation section), not generated text."),
        "Response time is processing latency for automatic answers only; human queue time, CSAT and repeat contacts cannot be measured offline.",
        "Segments with few tickets (e.g. 8 enterprise tickets in validation) make variation figures unstable.",
    ]
    return report


def training_overlap(items: list, train_path) -> dict:
    """How many input bodies are verbatim copies of the classifier's training tickets."""
    try:
        trained = {str(t.get("body") or "").strip().lower() for t in load_tickets(train_path) if isinstance(t, dict)}
    except (OSError, ValueError):
        return {"tickets": len(items), "exact_body_matches": "unknown (training file unreadable)"}
    bodies = [str(t.get("body") or "").strip().lower() for t in items if isinstance(t, dict)]
    return {"tickets": len(items), "exact_body_matches": sum(b in trained for b in bodies if b),
            "training_file": _repo_relative(train_path)}


def _repo_relative(path) -> str:
    """Repo-relative path for published results (never the local machine's absolute path)."""
    from pathlib import Path
    from . import ROOT
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def summary_markdown(m: dict) -> str:
    v, b, g = m["volume"], m["business"], m["governance"]
    lines = [f"# Evaluation summary — {m['run']['run_id']}", "",
             f"Input `{m['run']['input']}` ({m['run']['tickets_in_file']} tickets), engine `{m['run']['engine']}`, "
             f"classifier `{m['run']['classifier']}`, answer mode `{m['run'].get('answer_mode', 'extractive')}`, policy `{m['run']['policy_version']}`, "
             f"automatic sending {'ON' if m['run']['auto_send'] else 'OFF (kill switch)'}, {m['run']['date']}.", "",
             "| Measure | Value |", "|---|---|",
             f"| Processed / logged | {v['processed']} / {v['decisions_logged']} (reconciles: {g['decision_log_reconciles']}) |",
             f"| Auto-responded (sent) / held by kill switch | {v['auto_responded_sent']} / {v['auto_respond_decisions_held_by_kill_switch']} |",
             f"| Escalated / blocked | {v['escalated']} / {v['blocked_by_guardrails']} |",
             f"| FCR (actual / if released) | {b['first_contact_resolution_pct']}% / {b['first_contact_resolution_if_kill_switch_released_pct']}% |",
             f"| Escalation rate (actual / if released) | {b['escalation_rate_pct']}% / {b['escalation_rate_if_kill_switch_released_pct']}% |",
             f"| Latency median / p95 (s) | {m['technical']['latency_seconds']['median']} / {m['technical']['latency_seconds']['p95']} |",
             f"| Private data in outbound responses | {g['private_data_in_outbound_responses']} |",
             f"| Retriever (fallback) | {m['run']['retrieval']['name']}, {m['run']['retrieval']['chunking']} passages (fallback: {m['run']['retrieval']['fallback']}) |",
             f"| Cited quotes found at their recorded corpus span | {g['citations_resolving_to_corpus_text']} |"]
    if "generation" in m:
        q = m["generation"]
        lines += [f"| Generated answers (released or held) / cannot_answer / generation failed | {q['answers_released_or_held']} / {q['cannot_answer']} / {q['generation_failed']} |",
                  f"| Blocked by grounding checks / unsupported claims flagged | {q['blocked_by_grounding']} / {q['unsupported_claims_flagged']} |",
                  f"| Citations per answer / citation support rate (automated proxy) | {q['citations_per_answer']} / {q['citation_support_rate_automated_proxy']} |"]
    if "routing" in m:
        t, r = m["technical"], m["routing"]
        lines += [f"| Intent accuracy / macro precision / macro recall | {t['intent_accuracy']} / {t['intent_macro_precision']} / {t['intent_macro_recall']} |",
                  f"| Retrieval hit rate (any / top-1) | {t['retrieval_hit_rate_any_expected_doc']} / {t['retrieval_hit_rate_top1']} |",
                  f"| Citation accuracy vs expected docs (proxy) | {t['citation_accuracy_vs_expected_docs']} |",
                  f"| Auto precision / recall vs expected route | {r['auto_precision_vs_expected_route']} / {r['auto_recall_vs_expected_route']} |",
                  f"| must_not_auto_respond violations | {r['must_not_auto_respond_violations']} of {r['must_not_auto_respond_tickets']} |",
                  f"| Calibration max gap (points) | {g['confidence_calibration']['max_abs_gap_points']} |"]
    lines += ["", "**Limitations.** " + " ".join(m["limitations"])]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the CloudServe evaluation unattended and write a metrics report")
    p.add_argument("--input", required=True, help="any JSON list of tickets with the dataset schema")
    p.add_argument("--output", default="evaluation/results", help="output directory")
    add_engine_arguments(p)
    a = p.parse_args(argv)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    items = load_tickets(a.input)
    engine = build_engine(a)
    db = out / "decisions.sqlite"
    if db.exists():
        db.unlink()                 # each evaluation run is self-contained and reconcilable
    run_id = new_run_id()
    t0 = time.perf_counter()
    results, logged = run_batch(engine, items, db, run_id)
    wall = time.perf_counter() - t0
    run_info = {"run_id": run_id, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "input": str(a.input), "engine": engine.implementation,
                "classifier": f"{engine._models()['classifier']['name']} ({engine._models()['classifier']['version']})",
                "evidence_selector": engine._models()["evidence_selector"]["name"],
                "answer_mode": engine.answer_mode,
                "generator": (engine._models().get("generator") or {}).get("name"),
                "prompt_versions": engine._prompt_versions(), "policy_version": engine.policy["policy_version"],
                "auto_send": engine.auto_send, "retrieval": engine.retrieval_info,
                "training_overlap": training_overlap(items, a.train)}
    metrics = evaluate(items, results, logged, wall, run_info, kb=engine.kb)
    (out / "decisions.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "metrics_summary.md").write_text(summary_markdown(metrics), encoding="utf-8")
    write_human_review_sample(items, results, engine.kb, out / "human_review_sample.csv")
    v = metrics["volume"]
    print(f"Processed {v['processed']}/{len(items)}; logged {logged}; sent {v['auto_responded_sent']}; "
          f"held {v['auto_respond_decisions_held_by_kill_switch']}; escalated {v['escalated']}; "
          f"blocked {v['blocked_by_guardrails']}; metrics: {out / 'metrics.json'}")
    if not metrics["governance"]["decision_log_reconciles"]:
        raise SystemExit("Batch integrity check failed: decisions logged do not match tickets processed")


if __name__ == "__main__":
    main()
