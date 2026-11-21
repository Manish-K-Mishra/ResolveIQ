# CloudServe ticket resolution — Stage 5 baseline (Slices 1–4, remediated)

A LangGraph workflow that reads a support ticket from any of four channels, classifies its
intent, retrieves approved CloudServe documentation, and then either answers it with a
cited, verbatim documentation excerpt, escalates it to a person, or blocks it. Every
decision is written to a SQLite audit log with the Governance §2 minimum record.

Critical decisions (routing, safety, guardrails) are **deterministic Python** with a versioned
policy file. Models are used only behind adapters, for classification and evidence selection.
Automatic sending is **off by default** (kill switch). Auto-response answers are held for an
agent until you pass `--auto-send`.

> Design, routing policy, guardrails, data findings and known limitations:
> [docs/architecture.md](docs/architecture.md).

## 1. Setup (Python 3.11+)

```bash
python -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt -c requirements.lock
python -m pip install -e . --no-deps            # installs the cloudserve package from src/
python scripts/fetch_models.py                  # embedding model (~90 MB) into storage/models/
python -m pytest -q                             # no network or API key needed
```

`scripts/fetch_models.py` downloads all-MiniLM-L6-v2 (ONNX export and tokenizer) from the
Hugging Face Hub at a pinned revision and checks the SHA-256 of each file. Running it again
re-checks the files without downloading them. It is the only step that needs the internet. The
default policy (RP-1.0) retrieves lexically, so the workflow itself runs without the model.
The model is needed by `scripts/compare_retrieval.py` and
`scripts/derive_routing_policy.py --search-retrievers`, which measure the embedding retriever,
by the embedding tests, and by any policy that selects it. Such a policy falls
back to lexical retrieval if the model is missing, and every record says so.

Run every command from the repository root. `requirements.txt` pins direct dependencies;
`requirements.lock` pins the full tree that was tested. The second install only registers the
`src/cloudserve` package (dependencies are already installed); it is an editable install, so
default paths (`data/`, `config/`, `prompts/`, `.env`) resolve to this checkout.

## 2. Workflow

```
ingest -> classify -> retrieve -> route --(answer)--> select_evidence -> validate -> finalize
                                        `--(escalate)-------------------------------------^
```

| Step | What it does | Code |
|---|---|---|
| ingest | Normalises email / chat / docs / forum tickets; drops `labels` and `history` (evaluation only) | `src/cloudserve/ingest.py` |
| classify | Offline naive Bayes trained on development tickets, or PR-01 through an LLM provider | `src/cloudserve/classify.py`, `src/cloudserve/providers.py` |
| retrieve | Ranked passages (`DOC-XXX#article`) with scores and character offsets from `data/documentation.json`; lexical over whole articles in RP-1.0 | `src/cloudserve/retrieve.py` |
| route | Policy RP-1.0 decides: restricted intents, injection or sensitive text, a confidence floor of 0.70, and a retrieval floor of 0.40 with an approved document for the intent | `src/cloudserve/route.py`, `config/routing_policy.json` |
| select_evidence | The complete resolution section of the best-matching approved article, cited as `DOC-XXX#resolution` (deterministic), or PR-04 quote selection through an LLM | `src/cloudserve/generate.py` |
| validate | Grounding, PII, instruction-integrity and scope checks before release | `src/cloudserve/guardrails.py` |
| finalize | Action `auto_respond` / `escalate` / `block`, kill switch, handoff, audit record | `src/cloudserve/graph.py`, `src/cloudserve/logging_store.py` |

The thresholds and the retrieval configuration are derived from development data only, with
grouped out-of-fold cross-validation. Regenerate them with
`python scripts/derive_routing_policy.py`, which produces the same file every time apart from
`derived_on`. `--search-retrievers` repeats the derivation for every retrieval configuration
and writes evidence to `evaluation/results/retrieval_comparison/policy_search.json`, leaving
the policy unchanged. `python scripts/compare_retrieval.py` reproduces the chunking and retriever
comparison in `evaluation/results/retrieval_comparison/`. The decision is explained in
[docs/architecture.md §11](docs/architecture.md#11-retrieval-decision).

## 3. Run it

```bash
# Four channels: three answered, one escalated (exposed credentials)
python -m cloudserve.cli --input examples/four_channels.json \
  --output output/four.json --db output/four.sqlite --auto-send

# Engineered guardrail ticket (prompt injection): blocked even with sending on
python -m cloudserve.cli --input examples/guardrail_ticket.json \
  --output output/guard.json --db output/guard.sqlite --auto-send

# Inspect the audit log
sqlite3 output/guard.sqlite "select ticket_id, action_taken, status, reason from decision_log"
```

Without `--auto-send` (or `CLOUDSERVE_AUTO_SEND=true`), auto-response decisions get status
`held_for_review` and carry a `handoff` for the agent.

## 4. Unattended evaluation (A9 / A10)

```bash
python -m evaluation.harness --input data/validation_tickets.json \
  --output evaluation/results/validation_offline_default
```

`python evaluation/harness.py ...` and `python -m cloudserve.evaluation ...` are the same
entry point. The harness accepts any JSON list of tickets with the dataset schema. It writes
`decisions.json`, a fresh `decisions.sqlite`, `metrics.json` and `metrics_summary.md`. It
reconciles the number of logged decisions against the number of tickets processed and exits
with a non-zero status if they differ. Labels are
used only for scoring, after decisions are made.

The tested results saved under `evaluation/results/` are listed below. Each run records its policy version in `metrics_summary.md`; regenerate the runs after a policy change.

| Run | Folder |
|---|---|
| Validation, offline, kill switch on | `validation_offline_default/` |
| Validation, offline, `--auto-send` | `validation_offline_autosend/` |
| First 10 validation tickets, live Ollama `gemma4:e2b` (classifier and evidence selector) | `validation10_llm_ollama_gemma4/` |
| Chunking × retriever comparison (development and validation) | `retrieval_comparison/` |

Read the limitations section in each `metrics_summary.md`. For example, 62 of the 80
validation bodies are exact copies of development tickets, so offline intent accuracy
overstates generalisation.

CI (`.github/workflows/ci.yml`) installs exactly as in §1, runs the tests, then runs this
harness on `data/validation_tickets.json` into `storage/ci_eval` (uploaded as a build artefact).
`storage/` is for runtime output and is git-ignored.

## 5. Model providers (optional)

The default classifier needs no model. To use an LLM, add `--classifier llm` and, if you
want, `--evidence-selector llm`. The output schema and policy are the same for every provider.

```bash
# Ollama (verified locally with gemma4:e2b). Check `ollama list` first.
python -m cloudserve.cli --input examples/one_ticket.json --output output/llm.json --db output/llm.sqlite \
  --classifier llm --evidence-selector llm --provider ollama --model gemma4:e2b

# llama.cpp OpenAI-compatible server (not verified in this build)
...  --classifier llm --provider llamacpp --base-url http://localhost:8080/v1 --model local-model

# Anthropic Claude (verified on 10 tickets with claude-haiku-4-5; needs your own key; costs about $0.002 per ticket)
...  --classifier llm --provider anthropic --model claude-haiku-4-5
```

Claude setup: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` there locally. Never
commit `.env` (it is in `.gitignore`) or paste a key into a chat or agent prompt.
`CLOUDSERVE_MODEL` applies only to the provider named in `CLOUDSERVE_PROVIDER`. A missing key
stops the run with a clear message and no traceback.

The confidence floor of 0.70 was validated for the offline classifier only. LLM confidence is
an uncalibrated model score. Records mark this with
`thresholds.confidence_floor_validated_for_this_classifier`.

## 6. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CLOUDSERVE_AUTO_SEND` | `false` | Kill switch. `true` releases automatic answers. |
| `CLOUDSERVE_PROVIDER` / `CLOUDSERVE_MODEL` | `ollama` / provider default | LLM backend |
| `CLOUDSERVE_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `CLOUDSERVE_MAX_RETRIES` | `2` | Retries per call |
| `CLOUDSERVE_CALL_DEADLINE_SECONDS` | timeout × (retries+1) + 5 | Wall-clock cap per model call (covers slow streaming) |
| `OLLAMA_BASE_URL`, `LLAMACPP_BASE_URL` | localhost | Local servers |
| `CLOUDSERVE_EMBEDDING_MODEL_DIR` | `storage/models/all-MiniLM-L6-v2` | Embedding model files (only used by a policy that selects the embedding retriever) |

After 3 consecutive provider failures the circuit breaker opens. The remaining tickets in that
run then escalate immediately, with reason code `classification_model_failed`.

## 7. Repository layout

| Path | Contents |
|---|---|
| `src/cloudserve/` | Application code, one module per component (table in §2) |
| `prompts/build/`, `prompts/README.md` | Versioned prompts that run in the system, and the register |
| `prompts/evaluation/` | Prompts used to judge output (none yet; evaluation is deterministic) |
| `tests/` | Test suite (`python -m pytest -q`) |
| `evaluation/harness.py`, `evaluation/results/` | Evaluation entry point and saved runs |
| `config/routing_policy.json`, `scripts/` | Routing policy and the scripts that derive it, compare retrieval and fetch the model |
| `docs/architecture.md` | Design notes |
| `data/`, `examples/` | Supplied dataset files and small demo inputs |
| `.github/workflows/ci.yml` | Continuous integration |

## 8. Data and attribution

All inputs under `data/` come from the IIT Roorkee / Futurense capstone pack. This code was
produced with AI assistance and reviewed by independent reviewer passes. The full AI-use
declaration is in the project report. No API keys or customer credentials are included.
