# Incident response (Governance Framework §5)

**Use this when** a customer, an agent or a dashboard reports a wrong, unsafe or leaked automatic
answer. Run every command from the repository root with the project virtualenv active
(`source .venv/bin/activate`). You do not need to understand the code: follow the steps in order.

**Where things are**

| Thing | Location |
|---|---|
| Kill switch flag | `storage/KILL_SWITCH`. Override the path with `CLOUDSERVE_KILL_SWITCH_FILE`. |
| Decision log, API service | `storage/decisions.sqlite` (`--db` or `CLOUDSERVE_DB` changes it). `GET /health` shows the path in use under `decision_log`. |
| Decision log, CLI | `output/decisions.sqlite` unless `--db` was given |
| Decision log, evaluation harness | `<output dir>/decisions.sqlite` |
| Routing policy | `config/routing_policy.json`, versioned in git |
| Live state | `curl -s http://127.0.0.1:8000/health` and `curl -s http://127.0.0.1:8000/metrics` |

All times in the log are UTC ISO-8601 strings (`created_at`, `timestamp`), so a plain string
comparison such as `created_at >= '2026-11-21T19:00:00'` selects "since time T".

## The six steps

| Step | What to do (exact commands) | Who does it | How long it should take |
|---|---|---|---|
| **1. Detect** | Signals: a customer or agent complaint; Grafana panels for guardrail blocks, provider failures or cost; an unexpected `answers_total{delivery="sent"}` rate. Confirm what the system is doing now: `curl -s http://127.0.0.1:8000/health` (check `answers_released`, `kill_switch.engaged`, `policy_version`, `breaker`), then `curl -s http://127.0.0.1:8000/metrics \| grep -E 'answers_total\|guardrail_blocks_total\|kill_switch_engaged'`. Write down the report time as **T** in UTC. | On-call operations (anyone on the rota) | 5 minutes |
| **2. Contain** | **Engage the kill switch first, then investigate.** `python -m cloudserve.killswitch on --reason "INC: wrong billing answers, reported 02:10"`. Check it: `python -m cloudserve.killswitch status` prints `"engaged": true`, and `/health` shows `"answers_released": false`. No restart or deployment is needed. Every ticket that finalises from now on is held for an agent. If a hosted provider is suspected of leaking data, also restart the service with the offline classifier (`python -m cloudserve.api --classifier offline`). | On-call operations. If they cannot be reached, the support operations lead. | 1 minute (the switch takes effect at the next ticket) |
| **3. Assess** | Find every answer sent since T: `sqlite3 -header -column storage/decisions.sqlite "SELECT ticket_id, json_extract(record_json,'$.prediction.value') AS intent, json_extract(record_json,'$.citations[0]') AS cited, policy_version, created_at FROM decision_log WHERE answer_sent=1 AND created_at >= 'T' ORDER BY created_at;"`. Count them by outcome: `sqlite3 storage/decisions.sqlite "SELECT action_taken, status, count(*) FROM decision_log WHERE created_at >= 'T' GROUP BY 1,2;"`. Find every sent answer that used a suspect document: `sqlite3 storage/decisions.sqlite "SELECT d.ticket_id, d.created_at FROM decision_log d, json_each(d.record_json,'$.sources_used') s WHERE d.answer_sent=1 AND json_extract(s.value,'$.doc_id')='DOC-BILL-001';"`. Read one full record, including its reason, guardrail results, prompt version and requirement ids: `sqlite3 storage/decisions.sqlite "SELECT record_json FROM decision_log WHERE ticket_id='VAL-0008';" \| python -m json.tool`. Decide which of three causes it is: a document error (the cited passage is wrong), a routing error (the policy let through a ticket it should not have), or a generation error (`answer_mode` is `generated` and `unsupported_claims` is not empty). | Support operations lead, with the engineering lead | 30–60 minutes |
| **4. Notify** | Send each affected customer (the ticket_id list from step 3) a correction from a human agent. If private data went out, tell the security and compliance reviewer immediately; they decide on any regulatory notice. Post the summary in the incident channel: count sent since T, cause, whether the switch is on. | Support operations lead for customers. Security and compliance reviewer for data incidents. | Within 1 hour of containment for affected customers. Data incidents follow the compliance reviewer's timeline. |
| **5. Remediate** | **Document error:** the documentation owner corrects the article. Until it is fixed, remove that document from `intent_documents` in the policy (this is a new policy version; see below). **Routing error, policy rollback:** `git log --oneline -- config/routing_policy.json` lists the policy versions. Restore the previous one with `git show <commit>:config/routing_policy.json > storage/routing_policy_previous.json`, then restart with `python -m cloudserve.api --policy storage/routing_policy_previous.json`. The policy is read at start-up, so a restart is required. `/health` then shows the restored `policy_version`. A new policy is derived only from development data (`python scripts/derive_routing_policy.py`) and needs the project owner's approval. **Generation error:** run with `--answer-mode extractive`, the verbatim-documentation default. Add a regression test for the failing ticket before any fix is merged. | Engineering lead (policy, code). Documentation owner (articles). | Same day for containment fixes. A new policy version needs a re-run of the evaluation gate before release. |
| **6. Review** | Release the switch only after the fix has passed the gate: `python -m cloudserve.evaluation --input data/validation_tickets.json --output evaluation/results/<incident_run>` shows 0 `must_not_auto_respond` violations and logged = processed. Then run `python -m cloudserve.killswitch off` and check that `/health` shows `"answers_released": true` (only if the service runs with `--auto-send`). Write a blameless review: timeline from T, the number of affected tickets (step 3 query), the cause, and the change made. Add a row to the [risk register](risk_register.md) if the risk was not listed. | Project owner chairs the review. The engineering lead writes it up. | Review within 5 working days. Releasing the switch is the project owner's decision. |

## The kill switch

| Question | Answer |
|---|---|
| What is the mechanism? | A runtime flag file, `storage/KILL_SWITCH` (`CLOUDSERVE_KILL_SWITCH_FILE` overrides the path). `python -m cloudserve.killswitch on --reason "..."` engages it, `off` releases it and `status` shows it. Any file at the path engages the switch, even an empty one. Separately, automatic sending is **off by default**: answers are released only with `--auto-send` or `CLOUDSERVE_AUTO_SEND=true`, and the flag file overrides both. |
| Who is authorised to use it? | Anyone with write access to the deployment's `storage/` directory: on-call operations and the support operations lead. File-system permission is the authorisation control. There is no separate user model. Engaging it needs no approval. Releasing it is the project owner's decision. |
| How long until it takes effect? | From the next ticket that finalises. `finalize` reads the flag for every ticket. There is no restart, no deployment and no cache. Measured offline decision time is 0.0012 s at p95 (`final_offline_default`). |
| What happens to tickets in flight? | A ticket that has not yet reached `finalize` is held, even if it was classified and routed before the switch went on. An `auto_respond` decision is recorded with `delivery: held_kill_switch` and the reason says when and why the switch was engaged. Decisions already finalised (and answers already sent) are not changed. Use the step 3 query to find them. Escalations and blocks are unaffected. |
| How is it tested? | `test_kill_switch_engaged_mid_batch_holds_later_tickets_even_with_auto_send`, `test_kill_switch_is_read_at_finalize_for_tickets_in_flight`, `test_any_file_at_the_flag_path_engages_the_switch`, `test_kill_switch_cli_on_status_off` (tests/test_reliability.py); `test_auto_send_releases_and_kill_switch_overrides_via_api` (tests/test_api.py); `test_kill_switch_off_holds_answer_for_review` (tests/test_ai_graph.py). CI runs them on every push. The `/metrics` gauge `kill_switch_engaged` and `/health` show the live state. |

**Drill.** The commands above were dry-run against a scratch flag file and the committed
`final_offline_autosend/decisions.sqlite`, with `-readonly`. The "answers since T" query returned
the sent answers with intent and cited passage, and the document query returned the 2 sent
answers that used `DOC-ACCT-001`. A timed drill with an operator unfamiliar with the system has
**not** been run. It is to be confirmed by the project owner before a pilot.
