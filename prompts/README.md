# Prompt register

Every prompt that runs inside the system is a versioned file under `build/`. The file's front
matter (`id`, `version`, `changes_from_previous`) is the authoritative record; this table is the
index. The `id` and `version` are written into every decision record as `prompt_version`
(for example `PR-01 v1.1`), so a logged decision can be traced to the exact prompt text.

| ID | Version | File | Purpose | Used by | Change notes |
|---|---|---|---|---|---|
| PR-01 | 1.1 | `build/PR-01_classification.md` | Classify a ticket into one of the 22 dataset intents or `unknown`, with urgency, confidence, alternatives and a reason (strict schema) | `src/cloudserve/classify.py` `LangChainClassifier` (`--classifier llm`) | v1.0 was proposed in Stage 3 but never implemented. v1.1 implements it with two deviations: urgency is low/medium/high only (the dataset has no "critical" label), and alternatives are allowed intent labels, not free text. |
| PR-02 | 1.0 | `build/PR-02_grounded_answer.md` | Write a short answer (at most 6 sentences) from the approved passages only. Each sentence cites passage ids and carries a verbatim quote. Set `cannot_answer` when the passages do not answer. No refunds, credits or timelines (strict schema) | `src/cloudserve/generate.py` `LangChainAnswerGenerator` (`--answer-mode generated`); checked by `guardrails.validate_generated` | First implemented version (Slice 7). Stage 3 proposed grounded generation as PR-02 but it was never built. |
| PR-04 | 1.0 | `build/PR-04_evidence_selection.md` | Select one verbatim quote and its `doc_id` from the retrieved articles; never write free text | `src/cloudserve/generate.py` `LangChainEvidenceSelector` (`--evidence-selector llm`) | First registered version (it existed in code before it was in the Stage 3 register). Slice 5 updated only a file path in the front matter; the prompt text is unchanged, so the version is unchanged. |

Other prompt ids in the Stage 3 workbook are not in this register because no code runs them.
Grounded generation (PR-02) runs only in `--answer-mode generated`. The default, extractive mode
releases verbatim documentation excerpts. In both modes, routing and the release checks are
deterministic Python (`route.py`, `guardrails.py`).

The default run uses no prompt at all (offline classifier, extractive answer mode). Prompts are loaded from this repository folder at run time
(`cloudserve.providers.load_prompt`), so an edit here takes effect on the next run. Change a
prompt by raising its `version` and recording the reason in `changes_from_previous` and in this
table.

`evaluation/` holds prompts used to judge output. There are none yet; see
[evaluation/README.md](evaluation/README.md).
