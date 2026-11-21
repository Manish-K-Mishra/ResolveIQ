# Evaluation prompts

None are used yet. Evaluation is deterministic: `python -m evaluation.harness` scores decisions
against the dataset labels (intent, expected route, expected documents) with plain Python, and
no model judges any output. If a model-graded check is added later (for example a
claim-by-claim citation review), its prompt goes here and is registered in `../README.md`.
