"""Unattended evaluation harness (A9, A10): runs an evaluation set end to end.

    python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/<run>
    python evaluation/harness.py  --input ... --output ...       (same thing)

This is a thin entry point over cloudserve.evaluation.main, so the harness, the CLI and
`python -m cloudserve.evaluation` all run the same decision engine. It exits non-zero if the
decision log does not reconcile with the tickets processed.
"""
from cloudserve.evaluation import main

if __name__ == "__main__":
    main()
