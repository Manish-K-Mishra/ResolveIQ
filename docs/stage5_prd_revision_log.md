# Stage 5 PRD revision log

**Draft: the owner transfers this into the Stage 5 workbook.** It follows the sections of the
Stage 5 PRD Revision Log template. Version one is `docs/Stage_2_PRD_v1.docx` (with the Stage 3
prompt library). Every "now" statement describes code in this repository. Every figure links to a
committed result. Agreement and sign-off are **to be confirmed by the project owner**.

## 1. Revision summary

| Field | Value |
|---|---|
| New version number | 2.0 |
| Date of revision | 2026-11-21 to 2026-11-22 (one date per entry below) |
| Who carried out the revision | Engineering lead (build team) |
| Who reviewed and agreed it | To be confirmed by the project owner |
| Number of requirements changed | 5: FR-02, FR-03, FR-04, FR-05 and FR-09, plus the Stage 3 prompt register |
| Number of requirements added | 1 proposed: FR-13, the kill switch and release control |
| Number of requirements removed | 0. One design rule (blanket enterprise escalation) was removed. It was not a PRD requirement. |

## 2. The changes

| Requirement ID | What it said in version one | What it says now | What prompted the change (evidence) | Date | Agreed by |
|---|---|---|---|---|---|
| FR-04 (high-risk exclusions) | "Route … using an explicit, versioned policy: classification confidence, evidence quality, urgency and high-risk exclusions." The first build added a blanket rule: every enterprise ticket escalates. | Customer tier is **not** a routing input. High-risk exclusions are restricted intents, sensitive matter and injection only (RP-1.0). | Development labels mark 51 of 83 enterprise tickets as answerable, and no PRD requirement or Stage 1 finding asked for tier escalation. The rule was a fairness gap built in by design. Test: `test_enterprise_tier_is_not_escalated_by_tier_alone`. [architecture.md §3](architecture.md) | 2026-11-21 | To be confirmed by the project owner |
| FR-02 / PR-01 | PR-01 v1.0: urgency is `low\|medium\|high\|critical`. FR-04 names urgency as a routing input. | PR-01 v1.1: urgency is `low\|medium\|high` only, enforced by a strict schema. Alternatives must be allowed intent labels. Urgency comes from a transparent keyword rule and **does not drive routing**. | The dataset has no `critical` label. Urgency differs for identical bodies in 67 of 78 duplicate groups, so it cannot be learned from the text. Tests: `test_schema_rejects_free_text_intent_and_critical_urgency`, `test_pr01_prompt_is_loaded_versioned_and_delimits_untrusted_ticket`. [architecture.md §8](architecture.md), [prompts/README.md](../prompts/README.md) | 2026-11-21 | To be confirmed by the project owner |
| Stage 3 prompt register (PR-02, PR-04) | PR-02 (grounded answer) was proposed but not built. The evidence-selection prompt ran in code but was missing from the register. | PR-04 v1.0 is registered (evidence selection, `--evidence-selector llm`). PR-02 v1.0 is implemented (`--answer-mode generated`) and checked by `validate_generated`. Every record carries `prompt_version`. | Workbook and code contradicted each other. A decision must be traceable to the exact prompt. [prompts/README.md](../prompts/README.md) | 2026-11-21 (PR-04); 2026-11-22 (PR-02) | To be confirmed by the project owner |
| FR-04 (thresholds) | "Thresholds are not yet chosen: tune on development data." | RP-1.0 thresholds are **derived from development data only** by `scripts/derive_routing_policy.py`. It runs a 5-fold out-of-fold search grouped by ticket body, with the objective of maximum coverage subject to 0 `must_not_auto_respond` violations and auto precision ≥ 0.80. Result: confidence 0.70, retrieval 0.40. The policy file is versioned, and RP-1.x versions come only from this script. | 62 of 80 validation bodies duplicate development bodies, so tuning on validation would overfit the gate. Development out-of-fold: 205 auto answers, precision 0.805, 0 violations. Validation (80): 0.7879, 0 of 14 violations. RP-1.1 (windows) was evaluated and rejected. Test: `test_routing_policy_reproduces_from_development_data`. [architecture.md §3, §11](architecture.md), [final_offline_default](../evaluation/results/final_offline_default/metrics_summary.md) | 2026-11-21 (RP-1.0); 2026-11-22 (RP-1.1 rejected) | To be confirmed by the project owner |
| FR-09 and proposed FR-13 (release control) | FR-09: "Process a complete input dataset unattended … automatically save per-ticket outcomes." Nothing said whether answers are released. | **Automatic sending is off by default.** `auto_respond` decisions are held for an agent with a handoff. `--auto-send` or `CLOUDSERVE_AUTO_SEND=true` releases them. Proposed FR-13: a runtime kill switch (`python -m cloudserve.killswitch on`), read at finalize for every ticket, that overrides release with no deployment. | Governance §5 requires a kill switch. Auto precision (0.7879) is below the 0.80 objective on validation, and human quality review is pending. Tests: `test_kill_switch_off_holds_answer_for_review`, `test_kill_switch_is_read_at_finalize_for_tickets_in_flight`. [incident_response.md](governance/incident_response.md) | 2026-11-21 (default off); 2026-11-22 (runtime switch) | To be confirmed by the project owner |
| FR-05 (answer generation) | "Generate clear, appropriately scoped answers only from retrieved approved passages." Generation was assumed. | Two answer modes. **Extractive** (default) releases the verbatim resolution section of an approved article, with no generated claims. **Generated** (PR-02) writes at most 6 sentences, each citing passage ids with a verbatim quote, and blocks any sentence whose content-word overlap with its cited passages is below 0.6. | Extractive answers cannot hallucinate and need no model. Generated answers were measured: Claude Haiku 4.5 had auto precision 0.8947 but 10 grounding blocks. On PE review most were faithful paraphrases (proxy false positives) and about 3 were unsupported claims about the customer's situation (VAL-0005, VAL-0057, VAL-0071). [final_llm_anthropic_haiku45_generated](../evaluation/results/final_llm_anthropic_haiku45_generated/metrics_summary.md), [architecture.md §12](architecture.md) | 2026-11-22 | To be confirmed by the project owner |
| FR-03 (retrieval) | "Retrieve relevant passages … with a stable document/passage ID and relevance score." No method chosen. | RP-1.0 uses **lexical term overlap over whole articles**, with stable passage ids and character offsets. MiniLM embeddings are implemented but not the default. | Embeddings had better hit@1 (0.89–0.90 against 0.74) but no embedding configuration met the 0.80 precision objective with useful coverage. They retrieve near-miss articles for unanswerable tickets. A later finding: the lexical score dilutes on long tickets (36.5-point resolution gap, [fairness_audit.md](governance/fairness_audit.md)). [retrieval_comparison.md](../evaluation/results/retrieval_comparison/retrieval_comparison.md), [generation_comparison](../evaluation/results/generation_comparison/) | 2026-11-22 | To be confirmed by the project owner |

## 3. Assumptions that turned out to be wrong

| Assumption from version one | Did it hold? | What we found instead | What we changed because of it |
|---|---|---|---|
| Reviewed documents are sufficiently current for selected low-risk intents. | Cannot tell | All 29 documents have `last_reviewed_days_ago = 0`, so the data carries no freshness signal. | Answers may use only documents approved per intent (`intent_documents`). Freshness is recorded as an open risk (R-05). |
| The 500-ticket development mix approximates the relevant support categories. | Partly | Development has only 215 unique bodies across 500 tickets, and 62 of 80 validation bodies are copies of development ones. | Thresholds use grouped out-of-fold derivation. Every report states the overlap. Validation intent accuracy is labelled inflated. |
| Expected routing labels are a useful starting point, not guaranteed truth. | Held, more strongly than expected | `expected_route` differs for identical bodies in 35 of 78 duplicate groups. | Routing follows intent, safety and evidence rules, not label imitation. Precision and violations are reported alongside accuracy. |
| The support process has a staffed escalation destination. | Not tested | Escalation is 58.8% offline (73.8% Ollama, 76.2% Claude), so the queue stays large. | Handoff records carry the context and reason. Queue staffing is to be confirmed by the project owner. |
| Hosted inference can be used on de-identified capstone content. | Held for the capstone | Tickets can contain emails, phone numbers or keys, and the provider agreement is not established. | Ticket text is PII-masked before any hosted call. The offline classifier stays the default. |
| Offline routing labels and satisfaction proxies are not equivalent to observed customer outcomes. | Held | No customer outcome or human review exists yet. | Hallucination and satisfaction are reported as "not measured" rather than estimated. |

## 4. What we decided not to change

| What looked wrong | Why we left it | What it would have cost to change | Revisit when? |
|---|---|---|---|
| The grounding proxy blocked 10 Claude answers, most of them faithful paraphrases | Retuning the 0.6 overlap on validation would overfit the gate. A false block is fail-safe: it goes to a human. | A new threshold derived on development data, plus a human-reviewed sample of blocks | After the two-assessor review of `human_review_sample.csv` |
| Validation auto precision of 0.7879 is below the 0.80 objective | Thresholds were set on development data (0.805 there). Moving them to fit validation would overfit. | Re-derivation plus a new policy version | When a held-out set is available |
| Lexical retrieval biases against long tickets (36.5 points) | Found in the final audit. Any change to the score alters routing and needs a new derived policy. | A length-robust score, re-derivation and a full re-evaluation | Next policy version (RP-1.2) |
| Calibration gap of 59.8 points on validation (21.9 on development) | The confidence floor, not calibrated probability, drives routing. | A calibration layer fitted out of fold | Before any confidence is shown to agents |
| LLM p95 latency over 3 s (5.1–5.7 s) | The offline default meets it (0.0012 s). The LLM modes are optional. | A faster model or provider | If an LLM mode becomes the default |

## 5. Reflection

| Question | Answer (draft) |
|---|---|
| What did we most misunderstand about the problem when we wrote version one? | We treated it as a generation problem, when it is a routing problem. The decisive questions were which tickets may be answered at all and from which documents. The data's duplicate bodies and inconsistent labels mattered more than model quality. |
| Which piece of discovery work would have caught it earlier? | Profiling the dataset for duplicates and label consistency in week one: 62 of 80 validation copies, and 35 of 78 conflicting routes. |
| What would we do differently if we started this project again on Monday? | Build the evaluation harness, the reconciled decision log and the fairness script first. Then derive the policy from development data before adding any model. And segment results by ticket length from the start. |
| What is still uncertain, and what would we need to resolve it? | Real answer quality (needs the human review), behaviour on unseen tickets (needs a held-out set), behaviour under load (needs a load test), and whether the length and fluency gaps close with a better retrieval score (needs RP-1.2 and a re-run of the audit). |
