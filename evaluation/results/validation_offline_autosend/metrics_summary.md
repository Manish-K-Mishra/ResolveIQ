# Evaluation summary — run-20261121T185923-dd44fb

Input `data/validation_tickets.json` (80 tickets), engine `langgraph`, classifier `naive_bayes_intent (nb-a8a09fec6990)`, policy `RP-1.0`, automatic sending ON, 2026-11-21T18:59:23+00:00.

| Measure | Value |
|---|---|
| Processed / logged | 80 / 80 (reconciles: True) |
| Auto-responded (sent) / held by kill switch | 33 / 0 |
| Escalated / blocked | 47 / 0 |
| FCR (actual / if released) | 41.2% / 41.2% |
| Escalation rate (actual / if released) | 58.8% / 58.8% |
| Latency median / p95 (s) | 0.0009 / 0.0012 |
| Private data in outbound responses | 0 |
| Retriever (fallback) | lexical:term-overlap, article passages (fallback: False) |
| Cited quotes found at their recorded corpus span | 1.0 |
| Intent accuracy / macro precision / macro recall | 0.975 / 0.9849 / 0.9849 |
| Retrieval hit rate (any / top-1) | 0.8113 / 0.7358 |
| Citation accuracy vs expected docs (proxy) | 0.9615 |
| Auto precision / recall vs expected route | 0.7879 / 0.5417 |
| must_not_auto_respond violations | 0 of 14 |
| Calibration max gap (points) | 59.8 |

**Limitations.** Training overlap: 62 of 80 ticket bodies in this input are exact copies of development tickets the offline classifier was trained on; intent accuracy on those measures recall of training data, not generalisation, so treat it with caution (docs/architecture.md §8). Urgency and expected_route labels vary for identical ticket text in the supplied data, which caps achievable urgency and routing accuracy. Citation accuracy here is a label-based proxy (cited document among expected_doc_ids); it is not the human claim-by-claim review the Evaluation Framework requires, and hallucination rate is not measured. Responses are extractive (a verbatim documentation section), not generated text. Response time is processing latency for automatic answers only; human queue time, CSAT and repeat contacts cannot be measured offline. Segments with few tickets (e.g. 8 enterprise tickets in validation) make variation figures unstable.
