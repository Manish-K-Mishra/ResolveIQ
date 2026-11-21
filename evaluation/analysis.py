"""Evaluation Framework §4 results tables from the final runs' metrics.json files.

    python evaluation/analysis.py            # -> evaluation/results/final_report_tables.md

Reads only files the unattended harness wrote (metrics.json, decisions.json) plus the labelled
ticket files for the fairness segments (scripts/fairness_audit.py). Nothing is typed in by hand:
every "Achieved" figure comes from a run, and a run that does not exist yet is listed as missing.
Deterministic for the same inputs and --version.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"
# (folder, label, ticket file, note). The development run is IN-SAMPLE for the offline classifier.
RUNS = [
    ("final_offline_default", "Validation 80, offline, kill switch on (headline gate)", "data/validation_tickets.json", ""),
    ("final_offline_autosend", "Validation 80, offline, --auto-send", "data/validation_tickets.json", ""),
    ("final_llm_ollama_gemma4_generated", "Validation 80, LLM classifier + generated answers, Ollama gemma4:e2b", "data/validation_tickets.json", ""),
    ("final_llm_anthropic_haiku45_generated", "Validation 80, LLM classifier + generated answers, Claude Haiku 4.5", "data/validation_tickets.json", ""),
    ("final_offline_development", "Development 500, offline — IN-SAMPLE (the offline classifier was trained on these tickets; not a held-out result)",
     "data/development_tickets.json", "in-sample"),
]


def _fairness():
    spec = importlib.util.spec_from_file_location("fairness_audit", ROOT / "scripts" / "fairness_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pct(v, digits=1):
    return "—" if v is None else f"{100 * v:.{digits}f}%"


def num(v, unit=""):
    return "—" if v is None else f"{v}{unit}"


def load_run(folder: Path):
    mp = folder / "metrics.json"
    if not mp.exists():
        return None
    m = json.loads(mp.read_text(encoding="utf-8"))
    d = json.loads((folder / "decisions.json").read_text(encoding="utf-8")) if (folder / "decisions.json").exists() else []
    return m, d


def trained_on_dev(m: dict) -> bool:
    """Only the offline classifier is trained on the development tickets."""
    return "naive_bayes" in str(m["run"].get("classifier", ""))


def confidence_note(m: dict, in_sample: bool) -> str:
    n = m["run"]["tickets_in_file"]
    ov = m["run"].get("training_overlap", {}).get("exact_body_matches")
    parts = [f"n={n}"]
    if in_sample:
        parts.append("in-sample: classifier trained on these tickets")
    elif isinstance(ov, int) and ov and trained_on_dev(m):
        parts.append(f"{ov}/{n} bodies duplicate the development tickets the offline classifier and thresholds came from")
    elif isinstance(ov, int) and ov:
        parts.append(f"{ov}/{n} bodies duplicate the development tickets the thresholds came from; "
                     "the LLM classifier was not trained on them")
    return "; ".join(parts)


def results_table(m: dict, fair: dict | None, in_sample: bool) -> list[str]:
    b, t, g, r = m["business"], m["technical"], m["governance"], m.get("routing", {})
    gen = m.get("generation") or {}
    conf = confidence_note(m, in_sample)
    per_class = t.get("intent_precision_recall_by_class", {})
    below = sorted(k for k, v in per_class.items() if v["support"] and v["precision"] is not None and v["precision"] < 0.85)
    ttfr = b["time_to_first_reply_automatic_seconds"]
    sup = gen.get("citation_support_rate_automated_proxy")
    worst = None
    if fair:
        worst = max(((f, k, s["variation_points"][k]) for f, s in fair["segments"].items()
                     for k in ("resolution_rate", "quality_routing_accuracy")), key=lambda x: x[2])
    fcr_actual, fcr_rel = b["first_contact_resolution_pct"], b["first_contact_resolution_if_kill_switch_released_pct"]
    rows = [
        ("First contact resolution", "42%", "60%",
         f"{fcr_rel}% if released ({fcr_actual}% actually sent)", f"Low-medium ({conf})",
         "Offline proxy: share of tickets the system would answer without a person. Kill switch on means 0 actually sent." if not m["run"]["auto_send"]
         else "Offline proxy: share answered without a person."),
        ("Mean time to first reply", "8 to 12 hrs", "< 5 min",
         (f"mean {ttfr['mean']} s / median {ttfr['median']} s for {ttfr['n']} sent answers" if ttfr["n"] else "no answers sent (kill switch on)"),
         "Low", "Processing time only; escalated tickets wait in the human queue, which cannot be measured offline."),
        ("Satisfaction proxy", "3.2 / 5", "4.0", "not measured", "—",
         "Needs the two-assessor rubric review of human_review_sample.csv (not yet filled)."),
        ("Escalation rate", "58%", "≤ 30%", f"{b['escalation_rate_if_kill_switch_released_pct']}% if released", f"Low-medium ({conf})",
         "Escalate + block decisions; restricted intents are never answered by policy."),
        ("Classification precision", "—", "85% (per class)",
         f"macro {pct(t.get('intent_macro_precision'))}; {len(below)} of {sum(1 for v in per_class.values() if v['support'])} classes below 85%",
         f"{'Low' if in_sample or (trained_on_dev(m) and m['run'].get('training_overlap', {}).get('exact_body_matches')) else 'Medium'} ({conf})",
         ("Below 85%: " + ", ".join(below) + "." if below else "No class below 85% on this set.") + " Per-class table in the appendix."),
        ("Hallucination rate", "—", "≤ 5%", "not measured (human review pending)" + (
            f"; automated proxy: {pct(1 - sup)} of citations not supported by the deterministic check" if sup is not None else ""),
         "—", "Requires two assessors on ≥ 50 responses (human_review_sample.csv)."
         + ("" if gen else " Extractive answers are verbatim documentation, so they contain no generated claims.")),
        ("Citation accuracy", "—", "95%",
         f"{pct(t.get('citation_accuracy_vs_expected_docs'))} (cited article among expected docs)"
         + (f"; automated support {pct(sup)}" if sup is not None else ""),
         f"Medium ({conf})", "Label proxy plus automated support check, not the human claim-by-claim check. "
         f"Quotes found at their recorded corpus span: {g.get('citations_resolving_to_corpus_text')}."),
        ("Latency p95", "—", "< 3 s", f"{t['latency_seconds']['p95']} s", f"Medium (n={t['latency_seconds'].get('n', m['run']['tickets_in_file'])})",
         "End to end per decision on this machine; LLM-cache replays excluded."),
        ("Private data occurrences", "—", "0", str(g["private_data_in_outbound_responses"]), "Medium",
         "Automated pattern scan of every outbound response; manual sample review pending."),
        ("Cross-group variation", "—", "< 5 pts", (f"{worst[2]} pts ({worst[0]}, {worst[1]})" if worst else "—"),
         "Low (small segments)", "Largest spread across the fairness segments; see docs/governance/fairness_audit.md."),
    ]
    out = ["| Measure | Baseline | Target | Achieved | Confidence in the figure | Notes |", "|---|---|---|---|---|---|"]
    out += [f"| {a} | {b_} | {c} | {d} | {e} | {f} |" for a, b_, c, d, e, f in rows]
    gov = [f"Governance conditions: decisions logged {g['decisions_logged']} / processed {g['tickets_processed']} "
           f"(reconciles: {g['decision_log_reconciles']}); must_not_auto_respond violations {r.get('must_not_auto_respond_violations')} "
           f"of {r.get('must_not_auto_respond_tickets')}; calibration max gap {g.get('confidence_calibration', {}).get('max_abs_gap_points')} points "
           f"(target within 5); auto precision vs expected route {r.get('auto_precision_vs_expected_route')}."]
    if gen:
        gov.append(f"Generated answers: {gen['answers_released_or_held']} released or held, {gen['cannot_answer']} cannot_answer, "
                   f"{gen['blocked_by_grounding']} blocked by grounding, {gen['generation_failed']} generation failures, "
                   f"{gen['citations_per_answer']} citations per answer.")
    u = m.get("llm_usage") or {}
    if u.get("model_calls"):
        proj = u.get("projected_monthly_cost_usd", {})
        gov.append(f"Model use: {u['model_calls']} calls ({u['live_calls']} live, {u['cache_hits']} cache hits), tokens in/out "
                   f"{u['input_tokens']} / {u['output_tokens']}, cost if all live ${u['estimated_cost_if_all_calls_live_usd']}, "
                   f"projected ${proj.get('cost_usd')} a month at {proj.get('tickets_per_month')} tickets.")
    return out + [""] + gov


def appendix(m: dict) -> list[str]:
    t, g = m["technical"], m["governance"]
    out = ["Per-class intent precision and recall:", "", "| Intent | Support | Predicted | Precision | Recall |", "|---|---|---|---|---|"]
    for k, v in t.get("intent_precision_recall_by_class", {}).items():
        out.append(f"| {k} | {v['support']} | {v['predicted']} | {num(v['precision'])} | {num(v['recall'])} |")
    out += ["", "Confusion matrix (actual intent → predicted counts; off-diagonal entries are errors):", "",
            "| Actual | Predicted (count) |", "|---|---|"]
    for a, row in t.get("intent_confusion_matrix", {}).items():
        out.append(f"| {a} | " + ", ".join(f"{'**' + p + '**' if p != a else p}: {c}" for p, c in row.items()) + " |")
    cal = g.get("confidence_calibration", {})
    out += ["", "Confidence calibration (stated confidence vs observed intent accuracy):", "",
            "| Band | n | Stated | Observed | Gap (points) |", "|---|---|---|---|---|"]
    out += [f"| {b['band']} | {b['n']} | {b['stated']} | {b['observed']} | {b['gap_points']} |" for b in cal.get("bands", [])]
    lat, lat_all = t["latency_seconds"], t.get("latency_seconds_including_cache_hits", {})
    out += ["", "Latency (seconds per decision):", "", "| Scope | n | Mean | Median | p95 |", "|---|---|---|---|---|",
            f"| Live decisions | {lat.get('n', '—')} | {lat.get('mean')} | {lat.get('median')} | {lat.get('p95')} |"]
    if lat_all:
        out.append(f"| Including cache replays | {lat_all.get('n')} | — | {lat_all.get('median')} | {lat_all.get('p95')} |")
    return out


def build(version: str, results_dir: Path = RESULTS) -> str:
    fa = _fairness()
    L = ["# Final report tables", "",
         "Generated by `python evaluation/analysis.py` from the harness outputs in `evaluation/results/final_*`. "
         "Every figure is read from a run's `metrics.json` or `decisions.json`; nothing is typed in by hand.", "",
         f"- System version: git `{version}` (the repository state these tables were generated from; each run also records its "
         "policy and prompt versions below).",
         "- Hidden evaluation set: **never available.** The 120-ticket hidden set is not distributed with the project pack "
         "(Dataset Guide), so it was run 0 times. The 80-ticket validation file is the gate used here.",
         "- Development figures are **in-sample** for the offline classifier and must not be read as held-out results.", "",
         "| Run | Folder | Date (UTC) | Tickets | Classifier | Answer mode | Policy | Prompts |", "|---|---|---|---|---|---|---|---|"]
    loaded = []
    for folder, label, tickets, note in RUNS:
        run = load_run(results_dir / folder)
        if run is None:
            L.append(f"| {label} | `{folder}` | missing at generation time | — | — | — | — | — |")
            continue
        m, d = run
        loaded.append((folder, label, tickets, note, m, d))
        ri = m["run"]
        L.append(f"| {label} | `{folder}` | {ri['date']} | {ri['tickets_in_file']} | {ri['classifier']} | "
                 f"{ri.get('answer_mode', 'extractive')} | {ri['policy_version']} | {', '.join(ri.get('prompt_versions', []))} |")
    for folder, label, tickets, note, m, d in loaded:
        fair = fa.audit(json.loads((ROOT / tickets).read_text(encoding="utf-8-sig")), d) if d else None
        L += ["", f"## {label}", "", *results_table(m, fair, note == "in-sample")]
    for folder, label, tickets, note, m, d in loaded:
        L += ["", f"## Appendix — {label}", "", *appendix(m)]
    head = next(((m, d, t) for f, _, t, _, m, d in loaded if f == "final_offline_default"), None)
    caveats = []
    if head:
        ov = head[0]["run"].get("training_overlap", {})
        caveats.append(f"{ov.get('exact_body_matches')} of the {ov.get('tickets')} validation tickets duplicate development "
                       "tickets, which affects the offline classifier (trained on them) and the routing thresholds (derived from them), "
                       "not the LLM classifiers")
        tiers = fa.audit(json.loads((ROOT / head[2]).read_text(encoding="utf-8-sig")), head[1])["segments"]
        small = sorted((v["tickets"], f"{g} ({f})") for f, s_ in tiers.items() for g, v in s_["groups"].items())[:2]
        caveats.append("segments are small (" + ", ".join(f"{name}: {n} tickets" for n, name in small) + ")")
    caveats += ["the expected_route and urgency labels are inconsistent for identical ticket text",
                "satisfaction, hallucination rate and real time to first reply need human or live measurement that has not been done",
                "the hidden set was never available"]
    L += ["", "## Caution", "", "The figures above should be treated with caution because " + "; ".join(caveats) + "."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(RESULTS / "final_report_tables.md"))
    ap.add_argument("--version", default=None, help="system version to state (default: git describe)")
    a = ap.parse_args(argv)
    version = a.version or subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT,
                                          capture_output=True, text=True).stdout.strip() or "unknown"
    Path(a.out).write_text(build(version), encoding="utf-8")
    print(f"written {a.out}")


if __name__ == "__main__":
    main()
