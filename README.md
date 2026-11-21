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
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 1. Setup (Python 3.11+)

```bash
python -m venv .venv
source .venv/bin/activate                       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt -c requirements.lock
python -m pytest -q                             # 122 tests, no network or API key needed
```

`requirements.txt` pins direct dependencies; `requirements.lock` pins the full tree that was tested.

## 2. Workflow

```
ingest -> classify -> retrieve -> route --(answer)--> select_evidence -> validate -> finalize
                                        `--(escalate)-------------------------------------^
```

| Step | What it does | Code |
|---|---|---|
| ingest | Normalises email / chat / docs / forum tickets; drops `labels` and `history` (evaluation only) | `cloudserve/pipeline.py` |
| classify | Offline naive Bayes trained on development tickets, or PR-01 through an LLM provider | `cloudserve/classifier.py`, `cloudserve/ai.py` |
| retrieve | Lexical search over `data/documentation.json` | `cloudserve/pipeline.py` |
| route | Policy RP-1.0 decides: restricted intents, injection or sensitive text, a confidence floor of 0.70, and a retrieval floor of 0.40 with an approved document for the intent | `cloudserve/routing.py`, `config/routing_policy.json` |
| select_evidence | Verbatim resolution section (deterministic), or PR-04 quote selection through an LLM | `cloudserve/grounding.py` |
| validate | Grounding, PII, instruction-integrity and scope checks before release | `cloudserve/grounding.py` |
| finalize | Action `auto_respond` / `escalate` / `block`, kill switch, handoff, audit record | `cloudserve/graph.py`, `cloudserve/audit.py` |

The thresholds are derived from development data only, with grouped out-of-fold
cross-validation. Regenerate them with `python scripts/derive_routing_policy.py`, which
produces the same file every time apart from `derived_on`.

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
python -m cloudserve.evaluation --input data/validation_tickets.json \
  --output evaluation/results/validation_offline_default
```

The harness accepts any JSON list of tickets with the dataset schema. It writes
`decisions.json`, a fresh `decisions.sqlite`, `metrics.json` and `metrics_summary.md`. It
reconciles the number of logged decisions against the number of tickets processed. Labels are
used only for scoring, after decisions are made.

The tested results saved under `evaluation/results/` are:

| Run | Folder |
|---|---|
| Validation, offline, kill switch on | `validation_offline_default/` |
| Validation, offline, `--auto-send` | `validation_offline_autosend/` |
| First 10 validation tickets, live Ollama `gemma4:e2b` (classifier and evidence selector) | `validation10_llm_ollama_gemma4/` |

Read the limitations section in each `metrics_summary.md`. For example, 62 of the 80
validation bodies are exact copies of development tickets, so offline intent accuracy
overstates generalisation.

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

After 3 consecutive provider failures the circuit breaker opens. The remaining tickets in that
run then escalate immediately, with reason code `classification_model_failed`.

## 7. Data and attribution

All inputs under `data/` come from the IIT Roorkee / Futurense capstone pack. This code was
produced with AI assistance and reviewed by independent reviewer passes. The full AI-use
declaration is in the project report. No API keys or customer credentials are included.
