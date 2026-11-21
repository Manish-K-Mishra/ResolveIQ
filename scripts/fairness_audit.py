"""Fairness audit (Governance Framework §3; NFR-06) from a finished evaluation run.

    python scripts/fairness_audit.py --run evaluation/results/final_offline_default \
        --tickets data/validation_tickets.json --out evaluation/results/fairness/final_offline_default

Reads the run's decisions.json and the labelled ticket file (labels are used here only, after
every decision was made). Writes <out>.json and <out>.md. Deterministic: same inputs, same bytes.

Segments: customer tier, language fluency, ticket length (short = body word count at or below the
median of THIS input file, long = above it), channel and region.
Per segment:
  * resolution rate   = auto_respond decisions / tickets (first contact resolution if the kill
                        switch were released), with a Wilson 95% interval;
  * quality score     = routing accuracy vs the expected_route label (the decision matches the
                        label). This is a label proxy; human quality scores are pending the
                        two-assessor review sheet (human_review_sample.csv);
  * auto precision    = share of the segment's automatic answers whose label expected one;
  * citation support  = automated citation-support rate of generated answers (generated runs only);
  * variation from best, in percentage points, for resolution rate and quality score.
Investigation: for every segment, the resolution rate its INTENT MIX alone predicts (each intent's
run-wide automatic-answer rate, weighted by the segment's intents), the label ceiling (share of
the segment labelled auto_respond), intent accuracy and top-1 retrieval hit rate. If the
predicted rate tracks the actual one, the gap comes from what the segment asks about, not from
how the system treats it.
"""
from __future__ import annotations
import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_POINTS = 5.0
SEGMENTS = ("customer_tier", "language_fluency", "length", "channel", "customer_region")


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def frac(k, n):
    return round(k / n, 4) if n else None


def rel(path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def citation_support(r: dict) -> tuple[int, int]:
    s = [ok for e in (r.get("generation") or {}).get("sentences", []) for ok in e.get("citation_support", {}).values()]
    return (sum(s), len(s)) if r["action_taken"] == "auto_respond" else (0, 0)


def audit(tickets: list[dict], decisions: list[dict]) -> dict:
    by_id = {t["ticket_id"]: t for t in tickets if isinstance(t, dict) and isinstance(t.get("labels"), dict)}
    rows = [(by_id[r["ticket_id"]], r) for r in decisions if r["ticket_id"] in by_id]
    words = [len(str(t.get("body") or "").split()) for t, _ in rows]
    median_words = statistics.median(words) if words else 0
    auto = lambda r: r["action_taken"] == "auto_respond"
    intent_rate = {}
    for intent, grp in _group(rows, lambda t: t["labels"]["intent"]).items():
        intent_rate[intent] = sum(auto(r) for _, r in grp) / len(grp)

    def seg_value(t, field):
        if field == "length":
            return "short" if len(str(t.get("body") or "").split()) <= median_words else "long"
        return str(t.get(field))

    out = {}
    for field in SEGMENTS:
        groups = {}
        for g, grp in sorted(_group(rows, lambda t: seg_value(t, field)).items()):
            n = len(grp)
            k_auto = sum(auto(r) for _, r in grp)
            correct = sum((t["labels"].get("expected_route") == "auto_respond") == auto(r) for t, r in grp)
            auto_ok = sum(t["labels"].get("expected_route") == "auto_respond" for t, r in grp if auto(r))
            cs = [citation_support(r) for _, r in grp]
            cs_ok, cs_n = sum(a for a, _ in cs), sum(b for _, b in cs)
            with_docs = [(t, r) for t, r in grp if t["labels"].get("expected_doc_ids")]
            groups[g] = {
                "tickets": n,
                "resolution_rate": frac(k_auto, n), "resolution_rate_ci95": wilson(k_auto, n),
                "quality_routing_accuracy": frac(correct, n), "quality_routing_accuracy_ci95": wilson(correct, n),
                "auto_answers": k_auto, "auto_precision": frac(auto_ok, k_auto),
                "citation_support_rate_automated_proxy": frac(cs_ok, cs_n),
                "label_ceiling_expected_auto": frac(sum(t["labels"].get("expected_route") == "auto_respond" for t, _ in grp), n),
                "resolution_rate_predicted_by_intent_mix": round(sum(intent_rate[t["labels"]["intent"]] for t, _ in grp) / n, 4),
                "must_not_auto_respond_share": frac(sum(bool(t["labels"].get("must_not_auto_respond")) for t, _ in grp), n),
                "intent_accuracy": frac(sum(r["prediction"]["value"] == t["labels"]["intent"] for t, r in grp), n),
                "median_top_retrieval_score": round(statistics.median([r["sources_used"][0]["score"] if r["sources_used"] else 0.0
                                                                     for _, r in grp]), 4),
                # Why tickets labelled auto_respond were NOT answered, and why wrong answers were given.
                "missed_expected_auto_by_reason": dict(sorted(Counter(
                    _first_reason(r) for t, r in grp if t["labels"].get("expected_route") == "auto_respond" and not auto(r)).items())),
                "auto_answers_labelled_escalate": sum(t["labels"].get("expected_route") != "auto_respond" for t, r in grp if auto(r)),
                "retrieval_hit_top1": frac(sum(bool(r["sources_used"]) and r["sources_used"][0]["doc_id"] in t["labels"]["expected_doc_ids"]
                                               for t, r in with_docs), len(with_docs)),
            }
        for key in ("resolution_rate", "quality_routing_accuracy"):
            vals = [v[key] for v in groups.values() if v[key] is not None]
            best = max(vals) if vals else None
            for v in groups.values():
                v[f"{key}_points_below_best"] = round(100 * (best - v[key]), 1) if best is not None and v[key] is not None else None
        spread = {key: round(100 * (max(v[key] for v in groups.values()) - min(v[key] for v in groups.values())), 1)
                  for key in ("resolution_rate", "quality_routing_accuracy", "resolution_rate_predicted_by_intent_mix")}
        out[field] = {"groups": groups, "variation_points": spread,
                      "meets_under_5_points": {k: spread[k] < TARGET_POINTS for k in ("resolution_rate", "quality_routing_accuracy")}}
    largest = max(((f, k, out[f]["variation_points"][k]) for f in SEGMENTS for k in ("resolution_rate", "quality_routing_accuracy")),
                  key=lambda x: (x[2], x[0], x[1]))
    return {"tickets": len(rows), "length_split_median_body_words": median_words,
            "target_variation_points": TARGET_POINTS, "segments": out,
            "largest_gap": {"segment": largest[0], "measure": largest[1], "points": largest[2]}}


def _first_reason(r: dict) -> str:
    codes = [c for c in r.get("reason_codes", []) if c != "eligible_for_automatic_answer"]
    return codes[0] if codes else r["action_taken"]


def _group(rows, key):
    g = defaultdict(list)
    for t, r in rows:
        g[key(t)].append((t, r))
    return g


def pct(v):
    return "—" if v is None else f"{100 * v:.1f}%"


def markdown(report: dict, run_dir: str, tickets_path: str, run_info: dict) -> str:
    L = [f"# Fairness audit — {run_dir}", "",
         f"Run `{run_info.get('run_id')}` ({run_info.get('date')}), input `{tickets_path}`, {report['tickets']} labelled tickets, "
         f"classifier `{run_info.get('classifier')}`, answer mode `{run_info.get('answer_mode', 'extractive')}`, "
         f"policy `{run_info.get('policy_version')}`. Generated by `scripts/fairness_audit.py`.", "",
         "Resolution rate = automatic-answer decisions / tickets (first contact resolution if the kill switch were released). "
         "Quality score = routing accuracy against the `expected_route` label (a label proxy; human quality scores are pending the "
         "two-assessor review of `human_review_sample.csv`). Intervals are Wilson 95%. "
         f"Short tickets have at most {report['length_split_median_body_words']:g} body words (the median of this input).", ""]
    for field in SEGMENTS:
        s = report["segments"][field]
        L += [f"## {field}", "",
              "| Segment | Tickets | Resolution rate (95% CI) | Quality: routing accuracy (95% CI) | Points below best (resolution / quality) | Auto precision | Citation support (automated proxy) | Predicted by intent mix | Label ceiling | Intent accuracy | Retrieval hit@1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for g, v in s["groups"].items():
            ci = lambda c: "—" if not c else f"{100 * c[0]:.0f}–{100 * c[1]:.0f}"
            L.append(f"| {g} | {v['tickets']} | {pct(v['resolution_rate'])} ({ci(v['resolution_rate_ci95'])}) | "
                     f"{pct(v['quality_routing_accuracy'])} ({ci(v['quality_routing_accuracy_ci95'])}) | "
                     f"{v['resolution_rate_points_below_best']} / {v['quality_routing_accuracy_points_below_best']} | "
                     f"{pct(v['auto_precision'])} | {pct(v['citation_support_rate_automated_proxy'])} | "
                     f"{pct(v['resolution_rate_predicted_by_intent_mix'])} | {pct(v['label_ceiling_expected_auto'])} | "
                     f"{pct(v['intent_accuracy'])} | {pct(v['retrieval_hit_top1'])} |")
        L += ["", "| Segment | Median top retrieval score | Labelled auto but not answered, by first reason | Answered but labelled escalate |", "|---|---|---|---|"]
        for g, v in s["groups"].items():
            miss = ", ".join(f"{k}: {n}" for k, n in v["missed_expected_auto_by_reason"].items()) or "none"
            L.append(f"| {g} | {v['median_top_retrieval_score']} | {miss} | {v['auto_answers_labelled_escalate']} |")
        vp = s["variation_points"]
        L += ["", f"Variation: resolution {vp['resolution_rate']} pts, quality {vp['quality_routing_accuracy']} pts "
                  f"(target under {report['target_variation_points']:g}: "
                  f"{'met' if s['meets_under_5_points']['resolution_rate'] else 'NOT met'} / "
                  f"{'met' if s['meets_under_5_points']['quality_routing_accuracy'] else 'NOT met'}); "
                  f"intent mix alone predicts {vp['resolution_rate_predicted_by_intent_mix']} pts of resolution-rate spread.", ""]
    lg = report["largest_gap"]
    L += [f"Largest gap: `{lg['segment']}` {lg['measure']} at {lg['points']} points.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="evaluation output folder containing decisions.json and metrics.json")
    ap.add_argument("--tickets", default=str(ROOT / "data/validation_tickets.json"))
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args(argv)
    decisions = json.loads((Path(a.run) / "decisions.json").read_text(encoding="utf-8"))
    tickets = json.loads(Path(a.tickets).read_text(encoding="utf-8-sig"))
    run_info = {}
    mp = Path(a.run) / "metrics.json"
    if mp.exists():
        run_info = json.loads(mp.read_text(encoding="utf-8")).get("run", {})
    report = audit(tickets, decisions)
    report = {"run": rel(a.run), "tickets_file": rel(a.tickets), "run_id": run_info.get("run_id"),
              "run_date": run_info.get("date"), **report}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(report, rel(a.run), rel(a.tickets), run_info), encoding="utf-8")
    print(f"written {rel(out.with_suffix('.json'))} and .md; largest gap {report['largest_gap']}")


if __name__ == "__main__":
    main()
