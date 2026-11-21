# Evaluation summary — run-20261121T174636-87ff9a

Input `evaluation/inputs/validation_first10.json` (10 tickets), engine `langgraph`, classifier `ollama:gemma4:e2b (PR-01 v1.1)`, policy `RP-1.0`, automatic sending OFF (kill switch), 2026-11-21T17:46:52+00:00.

| Measure | Value |
|---|---|
| Processed / logged | 10 / 10 (reconciles: True) |
| Auto-responded (sent) / held by kill switch | 0 / 1 |
| Escalated / blocked | 9 / 0 |
| FCR (actual / if released) | 0.0% / 10.0% |
| Escalation rate (actual / if released) | 100.0% / 90.0% |
| Latency median / p95 (s) | 1.1265 / 5.7444 |
| Private data in outbound responses | 0 |
| Intent accuracy / macro precision / macro recall | 0.4 / 1.0 / 0.4286 |
| Retrieval hit rate (any / top-1) | 0.8333 / 0.8333 |
| Citation accuracy vs expected docs (proxy) | None |
| Auto precision / recall vs expected route | 0.0 / 0.0 |
| must_not_auto_respond violations | 0 of 4 |
| Calibration max gap (points) | 60.0 |

**Limitations.** The figures should be treated with caution because 7 of 10 ticket bodies in this input are exact copies of development tickets the offline classifier was trained on, so intent accuracy on them measures recall of training data, not generalisation (docs/BASELINE_AUDIT.md M1). Urgency and expected_route labels vary for identical ticket text in the supplied data, which caps achievable urgency and routing accuracy. Citation accuracy here is a label-based proxy (cited document among expected_doc_ids); it is not the human claim-by-claim review the Evaluation Framework requires, and hallucination rate is not measured. Responses are extractive (a verbatim documentation section), not generated text. Response time is processing latency for automatic answers only; human queue time, CSAT and repeat contacts cannot be measured offline. Segments with few tickets (e.g. 8 enterprise tickets in validation) make variation figures unstable.
