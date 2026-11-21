# CloudServe ticket resolution — architecture notes

## 1. Purpose and boundary

The system reads a CloudServe support ticket from any of four channels and ends with exactly
one of three logged actions:

- **auto_respond:** a cited answer taken from approved documentation.
- **escalate:** handed to a person with a summary.
- **block:** a guardrail fired, so nothing is released.

Models are used only for classification and for evidence selection, and both sit behind
adapters. Every decision that affects customer safety is deterministic Python driven by a
versioned policy file.

## 2. Workflow (LangGraph)

```
          +--------+   +----------+   +----------+   +-------+
 ticket ->| ingest |-->| classify |-->| retrieve |-->| route |
          +--------+   +----------+   +----------+   +---+---+
                                                         |
                       answer +--------------------------+-------------+ escalate
                              v                                        |
                     +-----------------+   +----------+                |
                     | select_evidence |-->| validate |--+             |
                     +-----------------+   +----------+  |             |
                                                         v             v
                                                     +-----------------------+
                                                     | finalize: action,     |
                                                     | kill switch, handoff, |
                                                     | audit record          |
                                                     +-----------------------+
```

`src/cloudserve/graph.py` compiles this as a `StateGraph` with one conditional edge out of `route`.
An escalated ticket never calls the evidence model. Every path, including internal errors and
malformed input, ends in `finalize`.

All modules are in `src/cloudserve/` (named after the Setup Guide §08 components).

| Component | Module | Technology |
|---|---|---|
| Channel normalisation (email, chat, docs_comment, forum) | `ingest.py` | Python dataclass. `labels` and `history` are dropped, because they are evaluation-only. |
| Intent and urgency classification | `classify.py` (offline naive Bayes and the LLM `LangChainClassifier`) | Multinomial naive Bayes trained on development tickets, or PR-01 through LangChain with a Pydantic schema |
| Retrieval | `retrieve.py` `KnowledgeBase` | Lexical overlap over `data/documentation.json`. Whole articles are the unit (29 short articles). |
| Routing policy | `route.py`, `config/routing_policy.json` | Deterministic rules, policy RP-1.0 |
| Evidence selection and answer rendering | `generate.py` | A verbatim resolution section (deterministic), or PR-04 through LangChain |
| Guardrails | `guardrails.py` (patterns, `contains_injection`, `validate_response`), `graph.py` (confidence floor, kill switch) | Grounding, PII, instruction integrity, scope, confidence floor |
| Audit log | `logging_store.py` | SQLite. Each record holds the Governance minimum fields. |
| Model providers | `providers.py` | Ollama, llama.cpp (OpenAI-compatible) and Anthropic, all through LangChain; timeouts, retries, deadline, circuit breaker |
| Workflow and wiring | `graph.py`, `runtime.py`, `cli.py` | LangGraph `StateGraph`; shared engine construction; command line |
| Evaluation harness | `evaluation.py`, entry point `evaluation/harness.py` | Unattended run that writes metrics JSON and a Markdown summary |

Prompts are versioned files in `prompts/build/`, indexed in `prompts/README.md`. Runtime
artefacts belong in `storage/` (git-ignored). CI (`.github/workflows/ci.yml`) runs the tests and
the unattended validation gate on every push and pull request.

## 3. Routing policy RP-1.0

A ticket can be answered automatically only if **all** of these hold:

1. No untrusted instruction (prompt injection) appears in the ticket text. If one does, the action is **block**.
2. The ticket contains no sensitive matter: legal, breach, exposed credentials, refund or chargeback.
3. The intent is known and is not restricted. The restricted intents are `compliance_request`, `feature_request`, `security_incident` and `unclear_request`.
4. Classification confidence is at least **0.70**. A missing confidence counts as 0.
5. A retrieved document is approved for that intent and scores at least **0.40**.

Every failing rule is recorded as a reason code, and its plain-language reason is written into the record.

The two thresholds were derived from the development tickets only, by
`scripts/derive_routing_policy.py`. It runs a 5-fold out-of-fold grid search, grouped by ticket
body so that duplicates cannot leak across folds. The objective is to maximise auto-answer
coverage, subject to **zero** `must_not_auto_respond` violations and auto-answer precision of at
least **0.80**.

On development data, out of fold, this gives intent accuracy of 0.954, 205 auto answers out of
500, auto precision of 0.805 and 0 violations. The script reproduces the policy file exactly,
apart from the `derived_on` date.

Customer tier is **not** a routing input. Development labels mark 51 of 83 enterprise tickets as
answerable, and no requirement asks for tier-based escalation.

## 4. Kill switch and release

Automatic sending is **off by default**. An `auto_respond` decision is then held with status
`held_for_review` and a handoff for the agent. `--auto-send` or `CLOUDSERVE_AUTO_SEND=true`
releases answers. Guardrail blocks apply in both modes.

## 5. Guardrails

| Check | Fails when | Effect |
|---|---|---|
| Instruction integrity | Ticket text matches injection patterns. Text is normalised first: NFKC, zero-width characters removed, digit-for-letter swaps undone. | **block**. The PII-masked input is stored for review. |
| Scope | Sensitive or legal matters in the ticket, or promises such as "we have refunded" in the answer | escalate or block |
| Grounding | The quote is not verbatim from the cited, retrieved document, or the citation is invalid | **block** |
| PII | Email, phone, API-key, access-key or card-number shapes in the outbound answer | **block** |
| Confidence floor | Confidence is below 0.70 | escalate |

## 6. Failure handling

- Each model call has a per-request timeout, bounded retries and a **wall-clock deadline**. The deadline catches servers that stream slowly.
- After **3 consecutive failures** a circuit breaker opens, and the remaining tickets escalate immediately.
- Malformed rows, empty bodies, unknown channels, no retrieval hit and a missing API key all end in a logged escalation or a clean configuration error, never a crash in the middle of a batch.

## 7. Audit record

Each decision is persisted before the next ticket is processed. The record holds
`decision_id`, `timestamp`, `ticket_id`, `stage`, `input_summary` (PII-masked), `model`,
`prediction`, `alternatives`, `sources_used`, `threshold_applied`, `action_taken`, `reason`,
`guardrail_results`, `prompt_version` and `requirement_ids`. It also holds `policy_version`,
`status`, `delivery`, `handoff` and `run_id`. `persist` rejects incomplete records. Each
evaluation run reconciles the number of logged decisions with the number of tickets processed.

## 8. Data findings that shaped the design

| Finding (computed from `data/`) | Design consequence |
|---|---|
| 62 of 80 validation bodies are exact copies of development bodies. Development has only 215 unique bodies across 500 tickets. | Offline validation accuracy overstates generalisation. Thresholds come from grouped out-of-fold development data, and every metrics report computes and states the overlap. |
| Urgency differs for identical bodies in 67 of 78 duplicate groups in development data. | Urgency cannot be learned from the text, so routing does not depend on urgency. A transparent keyword rule supplies it. |
| `expected_route` differs for identical bodies in 35 of 78 groups. Intent is consistent in all 78. | Routing is driven by intent, safety rules and evidence, and routing accuracy has a ceiling. |

## 9. Measured results

Full figures are in `evaluation/results/*/metrics_summary.md`.

| Run | Intent accuracy | Auto answers | Auto precision | Must-not violations | p95 latency |
|---|---|---|---|---|---|
| Validation (80), offline | 0.975 (inflated by overlap) | 33 | 0.788 | 0 of 14 | 1.3 ms |
| First 10 validation, Ollama gemma4:e2b | 0.40 | 1 | 0.0 | 0 of 4 | 5.7 s |
| First 10 validation, Claude Haiku 4.5 | 0.90 | 2 | 0.50 | 0 of 4 | 3.05 s |

## 10. Known limitations

- **Answers are extractive.** Each answer is a verbatim documentation section with fixed wording; nothing is generated. Grounded generation (PR-02) is not built yet.
- **Retrieval is lexical,** and 70% of unanswerable validation tickets still retrieve a document. The intent→document map is what prevents those tickets from being answered.
- **Confidence calibration misses the 5-point target.** On development data, out of fold, the 0.6–0.8 band states 0.705 but observes 0.44. LLM confidence is not calibrated at all.
- **Injection detection is pattern-based,** so a paraphrased injection can pass. The released answer is still a verified documentation excerpt.
- **The circuit breaker never recovers within a run** (there is no half-open state). Ticket text is sent unmasked to hosted providers.
- **llama.cpp has not been run live,** and LLM runs are not reproducible without a response cache.
- **Auto-answer rates differ between groups:** fluency 12.7 points, tier 14.3, region 15.4. Group sizes are small, and the cause has not been investigated yet.
