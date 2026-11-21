"""Process one ticket or a JSON list of tickets and print the decisions summary.

    python -m cloudserve.cli --input examples/one_ticket.json --output output/one.json --db output/one.sqlite

The audit DB accumulates across runs (it is a persistent log); each run has its own run_id and
reconciliation is per run. For a full evaluation with metrics use `python -m cloudserve.evaluation`.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from .audit import new_run_id
from .pipeline import load_tickets
from .runtime import add_engine_arguments, build_engine, run_batch


def main(argv=None):
    p = argparse.ArgumentParser(description="Run the CloudServe decision workflow on tickets")
    p.add_argument("--input", required=True, help="JSON file: one ticket object or a list of tickets")
    p.add_argument("--output", default="output/decisions.json")
    p.add_argument("--db", default="output/decisions.sqlite")
    add_engine_arguments(p)
    a = p.parse_args(argv)
    engine = build_engine(a)
    run_id = new_run_id()
    results, logged = run_batch(engine, load_tickets(a.input), a.db, run_id)
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    statuses = Counter(r["status"] for r in results)
    print(f"Run {run_id}: processed {len(results)}; logged {logged}; " +
          ", ".join(f"{k} {v}" for k, v in sorted(statuses.items())) +
          f"; automatic sending {'ON' if engine.auto_send else 'OFF (kill switch)'}")
    print(f"Results: {target}; audit DB: {a.db}")
    if logged != len(results):
        raise SystemExit("Audit reconciliation failed: logged decisions do not match processed tickets")


if __name__ == "__main__":
    main()
