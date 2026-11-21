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
| Retrieval | `retrieve.py` `KnowledgeBase` | Ranked passages with stable ids and character offsets. Policy RP-1.0 uses lexical term overlap over whole articles (29 passages). all-MiniLM-L6-v2 embeddings (ONNX Runtime, in-memory index) were measured and are supported, with a lexical fallback. See §11. |
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
5. A retrieved passage belongs to a document approved for that intent and scores at least **0.40**.
   The score is lexical: the share of the ticket's terms found in the article. The retriever and
   chunking are recorded in the policy file under `retrieval`.

Every failing rule is recorded as a reason code, and its plain-language reason is written into the record.

The two thresholds were derived from the development tickets only, by
`scripts/derive_routing_policy.py`. It runs a 5-fold out-of-fold grid search, grouped by ticket
body so that duplicates cannot leak across folds. The objective is to maximise auto-answer
coverage, subject to **zero** `must_not_auto_respond` violations and auto-answer precision of at
least **0.80**.

On development data, out of fold, this gives intent accuracy of 0.954, 205 auto answers out of
500, auto precision of 0.805 and 0 violations. The script reproduces the policy file exactly,
apart from the `derived_on` date. RP-1.0 and its 0.80 objective are owner-approved.
`--search-retrievers` repeats the derivation for every retrieval configuration in §11 and
writes the result to `evaluation/results/retrieval_comparison/policy_search.json` as evidence.
It never changes the policy file.

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
| Grounding | The quote is not verbatim from the cited passage, the cited passage is not the resolution section of an approved retrieved article, or the passage id is malformed or names another article | **block** |
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
`status`, `delivery`, `handoff` and `run_id`. `sources_used` lists each retrieved passage
(`passage_id`, `doc_id`, `section`, `score`, `start`/`end` offsets). `citations` holds the cited
passage id, and `citation_spans` the exact character span of the quote in the article, so
`content[start:end]` is the quoted text. `retrieval` names the retriever that actually ran and
whether it was the fallback. `persist` rejects incomplete records. Each
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
- **Retrieval is lexical** in RP-1.0, and 70% of unanswerable validation tickets still retrieve a passage. The intent→document map is what prevents those tickets from being answered. Embeddings retrieve far better (§11), but no embedding configuration met the routing objective with useful coverage.
- **Confidence calibration misses the 5-point target.** On development data, out of fold, the 0.6–0.8 band states 0.705 but observes 0.44. LLM confidence is not calibrated at all.
- **Injection detection is pattern-based,** so a paraphrased injection can pass. The released answer is still a verified documentation excerpt.
- **The circuit breaker never recovers within a run** (there is no half-open state). Ticket text is sent unmasked to hosted providers.
- **llama.cpp has not been run live,** and LLM runs are not reproducible without a response cache.
- **Auto-answer rates differ between groups:** fluency 12.7 points, tier 14.3, region 15.4. Group sizes are small, and the cause has not been investigated yet.

## 11. Retrieval decision

Requirements (Build Specification, "Retrieve"): "Searches the supplied documentation corpus and
returns ranked passages with scores." "Returns identifiers that resolve to the real corpus, so
citations can be verified." "Applies a relevance threshold and returns nothing rather than
something irrelevant." "Chunking strategy is a decision you make, justify and are able to
explain." Setup Guide §04: embed with all-MiniLM-L6-v2, and "Try at least two configurations,
measure the difference in retrieval quality, and record which you chose and why."

**Passages.** Each passage has an id `DOC-XXX#<slug>` and `start`/`end` offsets into the
article's `content`, and `content[start:end]` is exactly the passage text
(`KnowledgeBase.resolve`). We measured three chunking configurations:

- **article**: one passage per article (29 passages).
- **section**: one passage per `## ` section, giving Symptoms, Common causes, Resolution and Notes (116).
- **window**: ~800 characters with 120 overlap, cut on word or line breaks, as in the Setup Guide (58).

The search passage and the answer unit are separate. Search runs on the configured passages.
The answer always quotes the **complete Resolution section** of the best-matching approved
article (`DOC-XXX#resolution`), so a window that cuts a numbered sequence is never quoted (the
Dataset Guide's warning).

**Retrievers.**

- **lexical**: the share of the ticket's terms found in the passage. The floor is 0.26 with at least two shared terms.
- **embedding**: cosine similarity of all-MiniLM-L6-v2 vectors. The floor is 0.35, the highest floor that loses ≤1% of answerable development tickets. Unrelated queries such as "What is the weather in Paris tomorrow?" score about 0.08–0.13.

The model runs on CPU through ONNX Runtime, using the model's official ONNX export and tokenizer
at a pinned revision. `scripts/fetch_models.py` fetches them into `storage/models/` and checks
their SHA-256. Nothing is downloaded at run time. Dependencies: `onnxruntime`, `tokenizers`,
`numpy`. We chose this over sentence-transformers (PyTorch, far larger) and fastembed (wraps the
same runtime with more dependencies).

The index is an in-memory matrix, not Chroma, for the same reason the Setup Guide gives for
SQLite over PostgreSQL: at about a hundred passages, an exact dot product takes microseconds and
needs no store, no persistence and no service. Chroma would only add a dependency and a cache to
keep in step with the corpus.

If a policy names the embedding retriever and the model cannot be loaded, the run uses the
policy's lexical fallback with that fallback's own thresholds. It warns on stderr, and every
record's `retrieval` block says `fallback: true` with the reason.

**Measured retrieval quality** (`python scripts/compare_retrieval.py`; full tables, a floor sweep
and validation figures are in `evaluation/results/retrieval_comparison/`). Development tickets,
document-level hits against `labels.expected_doc_ids`, at each retriever's floor:

| Chunking | Retriever | hit@1 | hit@3 | Empty (answerable) | Empty (unanswerable) | hit@3 fluent / non-fluent |
|---|---|---|---|---|---|---|
| article | lexical | 0.740 | 0.796 | 0.118 | 0.161 | 0.807 / 0.759 |
| section | lexical | 0.675 | 0.703 | 0.230 | 0.322 | 0.722 / 0.644 |
| window | lexical | 0.734 | 0.773 | 0.140 | 0.182 | 0.785 / 0.736 |
| article | embedding | 0.891 | 0.936 | 0.014 | 0.126 | 0.952 / 0.885 |
| section | embedding | 0.899 | 0.950 | 0.006 | 0.098 | 0.967 / 0.897 |
| window | embedding | 0.891 | 0.936 | 0.011 | 0.119 | 0.952 / 0.885 |

Embeddings find the right article far more often: +16 points hit@1, and almost no answerable
ticket goes empty. Retrieval is worse for non-fluent English in every configuration (5–8 points
hit@3), as the Governance Framework predicts. A lexical+embedding hybrid added under 1 point of
hit@1 and nothing at hit@3, so it was not built.

No retriever returns nothing for most unanswerable tickets. In this dataset they are on-topic
(for example a database question the documentation does not cover), so they resemble the corpus
in both words and meaning.

**Effect on routing, and the choice.** Retrieval quality is not the objective by itself. Routing
depends on the retrieval score, so `python scripts/derive_routing_policy.py --search-retrievers`
derived thresholds for each configuration with the RP-1.0 objective (§3, development
out-of-fold; `evaluation/results/retrieval_comparison/policy_search.json`):

| Chunking | Retriever | Meets objective | Thresholds | Auto answers | Auto precision |
|---|---|---|---|---|---|
| **article** | **lexical (RP-1.0)** | **yes** | **0.70 / 0.40** | **205** | **0.805** |
| section | lexical | yes | 0.50 / 0.45 | 54 | 0.907 |
| window | lexical | yes | 0.70 / 0.35 | 238 | 0.8025 |
| article | embedding | no | — | — | at most 0.786 |
| section | embedding | no | — | — | at most 0.759 |
| window | embedding | yes | 0.99 / 0.70 | 44 | 0.841 |

**Embedding: better hit rate, fails the precision objective.** The embedding retriever finds the
expected article far more often (hit@1 0.90 against 0.74). It still cannot meet the 0.80
auto-precision target with useful coverage. It retrieves confidently for tickets that are
labelled unanswerable but sit inside an answerable intent, such as a database question the
documentation does not cover. Of the 97 wrong automatic answers at section/embedding
(0.70 / 0.35), 83 are such tickets. Better retrieval surfaces exactly the near-miss articles that
the weaker lexical score happened to filter out.

**RP-1.1 (lexical, 800/120 windows) was evaluated and rejected.** On development data it gave
238 automatic answers at precision 0.8025, against 205 at 0.805. On validation (80) it gave 35
automatic answers against 33, but auto precision fell from 0.788 to 0.771. It was also picked
from a search space six times larger than the one that produced RP-1.0, which inflates
development results. That evidence does not justify replacing an owner-approved policy, so
**RP-1.0 (lexical, whole articles, 0.70 / 0.40) stays the default**.

**Next (Slice 7).** We will re-evaluate the embedding retriever together with generated answers
that carry an explicit `cannot_answer` check, because the missing piece is an answerability
decision rather than better ranking. The embedding retriever, the fallback and both
measurement scripts stay in the code and tests for that work.
