# Evaluation summary — run-20261121T180155-b5ee56

Input `evaluation/inputs/validation_first10.json` (10 tickets), engine `langgraph`, classifier `anthropic:claude-haiku-4-5 (PR-01 v1.1)`, policy `RP-1.0`, automatic sending OFF (kill switch), 2026-11-21T18:02:16+00:00.

| Measure | Value |
|---|---|
| Processed / logged | 10 / 10 (reconciles: True) |
| Auto-responded (sent) / held by kill switch | 0 / 2 |
| Escalated / blocked | 8 / 0 |
| FCR (actual / if released) | 0.0% / 20.0% |
| Escalation rate (actual / if released) | 100.0% / 80.0% |
| Latency median / p95 (s) | 1.8801 / 3.0487 |
| Private data in outbound responses | 0 |
| Intent accuracy / macro precision / macro recall | 0.9 / 1.0 / 0.8571 |
| Retrieval hit rate (any / top-1) | 0.8333 / 0.8333 |
| Citation accuracy vs expected docs (proxy) | 1.0 |
| Auto precision / recall vs expected route | 0.5 / 0.2 |
| must_not_auto_respond violations | 0 of 4 |
| Calibration max gap (points) | 55.0 |

**Limitations.** Training overlap: 7 of 10 ticket bodies in this input are exact copies of development tickets the offline classifier was trained on; intent accuracy on those measures recall of training data, not generalisation, so treat it with caution (docs/BASELINE_AUDIT.md M1). Urgency and expected_route labels vary for identical ticket text in the supplied data, which caps achievable urgency and routing accuracy. Citation accuracy here is a label-based proxy (cited document among expected_doc_ids); it is not the human claim-by-claim review the Evaluation Framework requires, and hallucination rate is not measured. Responses are extractive (a verbatim documentation section), not generated text. Response time is processing latency for automatic answers only; human queue time, CSAT and repeat contacts cannot be measured offline. Segments with few tickets (e.g. 8 enterprise tickets in validation) make variation figures unstable.
