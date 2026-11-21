# Decision logging (Governance Framework §1)

Every processed ticket produces **one decision record**. The record is written to the SQLite
decision log before the next ticket is processed (`logging_store.persist`). The full record is
stored in `decision_log.record_json`. The fields used for reconciliation and incident queries
are also stored as columns: `decision_id`, `run_id`, `ticket_id`, `channel`, `stage`,
`action_taken`, `status`, `answer_sent`, `policy_version`, `prompt_version`, `reason`,
`created_at`. `persist` refuses a record that lacks any of the minimum fields
(`logging_store.GOVERNANCE_FIELDS`).

## Mapping to the minimum record

| Governance §1 field | Our field | What it holds | Example (`final_offline_default`, first record) |
|---|---|---|---|
| `decision_id` | `decision_id` | A UUID4 hex string, also the table's primary key | `67a4127e2cd64834a5ae70ae6d2592c0` |
| `timestamp` | `timestamp` | When the decision was made (UTC ISO-8601). `started_at` and `completed_at` are also recorded, and `created_at` when the row was written. | `2026-11-21T19:57:10.933978+00:00` |
| `ticket_id` | `ticket_id` | The input ticket id | `VAL-…` |
| `stage` | `stage` | The stage that settled the outcome: `routing` for escalations and pre-generation blocks, `validation` when the answer reached the pre-send checks. All stages run for each ticket, and their outputs are in the same record, so there is one record per ticket rather than one per stage. | `routing` |
| `input_summary` | `input_summary` | Channel, tier and the PII-masked subject and body, truncated | `forum ticket from a business customer: Need to rotate an exposed key …` |
| `model` (name, version) | `model` | The name and version of every model that ran: the classifier, the evidence selector or the generator. The offline classifier version is a hash of its training data. | `naive_bayes_intent` / `nb-a8a09fec6990` |
| `prediction` (value, confidence) | `prediction` | Intent, confidence and urgency, plus `confidence_type` (which says what the number means) and `error_type` | `api_key_issue`, 1.0 |
| `alternatives` | `alternatives` | The next three intents with their confidence | `rate_limit` 0.0, … |
| `sources_used` (doc_id, score) | `sources_used` | Every retrieved passage: `passage_id`, `doc_id`, `section`, `score` and `start`/`end` character offsets. `citations` and `citation_spans` give the exact quoted span. | `DOC-AUTH-004`, 0.4286 |
| `threshold_applied` | `threshold_applied`, `thresholds` | The confidence floor applied, plus the retrieval floor and whether the floor was validated for this classifier | 0.7; retrieval 0.4 |
| `action_taken` | `action_taken` | `auto_respond`, `escalate` or `block`. `status` and `delivery` add whether the answer was sent, held for review, or held by the kill switch. | `escalate` |
| `reason` | `reason`, `reason_codes` | A plain-language sentence for each failed rule, plus machine-readable codes | "The ticket mentions a legal, breach, exposed-credential or refund matter, which only staff may handle." |
| `guardrail_results` | `guardrail_results`, `guardrail_failures` | `pii`, `grounding`, `instruction_integrity`, `scope` and `confidence_floor`, each `pass`, `fail` or `not_run`. Generated mode adds `unsupported_claims` and the per-sentence checks under `generation`. | `scope: fail` |
| `prompt_version` | `prompt_version` | Every prompt used, for example `PR-01 v1.1` and `PR-02 v1.0`. The offline path says explicitly that no prompt was used. | `none (no model prompt used: …)` |
| `requirement_ids` | `requirement_ids` | The PRD requirements that produced this decision | `FR-01 … FR-08` |

**Fields beyond the minimum:** `policy_version` (RP-1.0), `run_id`, `answer_mode`, `retrieval`
(the retriever that actually ran and whether it was the fallback), `kill_switch` (the state read
at finalize), `handoff`, `model_calls`, `usage` (tokens and cost), `llm_cache`,
`pii_masked_for_provider` and `latency_seconds`.

**Tests:** `test_every_path_writes_a_complete_governance_record` and
`test_persist_rejects_incomplete_records` (tests/test_ai_graph.py) check that every path
(answered, released, escalated, blocked, failing classifier, malformed row) writes all minimum
fields and that the log survives a restart.
`test_batch_continues_past_bad_rows_and_reconciles` (tests/test_ai_graph.py) and
`test_reconciliation_failure_exits_non_zero` (tests/test_repository_layout.py) check that the
coverage check is enforced.

## Coverage check: reconciliation on the final runs

The harness and the CLI count logged rows for the run against processed tickets. A mismatch
exits non-zero. Counts were re-checked directly from each run's SQLite log on 2026-11-21 (UTC)
with
`sqlite3 -readonly <run>/decisions.sqlite "select count(*), count(distinct ticket_id), sum(answer_sent), count(distinct run_id) from decision_log"`:

| Run | Tickets processed (`decisions.json`) | Rows logged | Distinct tickets | Answers sent | Runs in log | Reconciles |
|---|---|---|---|---|---|---|
| `final_offline_default` | 80 | 80 | 80 | 0 (kill switch on) | 1 | yes |
| `final_offline_autosend` | 80 | 80 | 80 | 33 | 1 | yes |
| `final_llm_ollama_gemma4_generated` | 80 | 80 | 80 | 0 | 1 | yes |
| `final_llm_anthropic_haiku45_generated` | 80 | 80 | 80 | 0 | 1 | yes |
| `final_offline_development` (in-sample) | 500 | 500 | 500 | 0 | 1 | yes |

Each run's `metrics.json` also records `decisions logged N / processed N (reconciles: True)`,
as reproduced in [final_report_tables.md](../../evaluation/results/final_report_tables.md).

## Answering "was this behaviour intended?"

Every row holds the `policy_version`, `prompt_version` and `requirement_ids` that produced it,
so a decision can be traced to the policy file in git, the prompt file in `prompts/build/`, and the
PRD requirement. The `sqlite3` queries in [incident_response.md](incident_response.md) step 3
retrieve records by time window, document or ticket.
