# Evaluation summary — run-20261121T191534-5a00d0

Input `data/validation_tickets.json` (80 tickets), engine `langgraph`, classifier `naive_bayes_intent (nb-a8a09fec6990)`, answer mode `generated`, policy `RP-1.0`, automatic sending OFF (kill switch), 2026-11-21T19:17:32+00:00.

| Measure | Value |
|---|---|
| Processed / logged | 80 / 80 (reconciles: True) |
| Auto-responded (sent) / held by kill switch | 0 / 31 |
| Escalated / blocked | 47 / 2 |
| FCR (actual / if released) | 0.0% / 38.8% |
| Escalation rate (actual / if released) | 100.0% / 61.2% |
| Latency median / p95 (s) | 0.0033 / 4.9358 |
| Private data in outbound responses | 0 |
| Retriever (fallback) | lexical:term-overlap, article passages (fallback: False) |
| Cited quotes found at their recorded corpus span | 1.0 |
| Generated answers (released or held) / cannot_answer / generation failed | 31 / 0 / 0 |
| Blocked by grounding checks / unsupported claims flagged | 2 / 9 |
| Citations per answer / citation support rate (automated proxy) | 1.903 / 0.9524 |
| Intent accuracy / macro precision / macro recall | 0.975 / 0.9849 / 0.9849 |
| Retrieval hit rate (any / top-1) | 0.8113 / 0.7358 |
| Citation accuracy vs expected docs (proxy) | 1.0 |
| Auto precision / recall vs expected route | 0.7742 / 0.5 |
| must_not_auto_respond violations | 0 of 14 |
| Calibration max gap (points) | 59.8 |

**Limitations.** Training overlap: 62 of 80 ticket bodies in this input are exact copies of development tickets the offline classifier was trained on; intent accuracy on those measures recall of training data, not generalisation, so treat it with caution (docs/architecture.md §8). Urgency and expected_route labels vary for identical ticket text in the supplied data, which caps achievable urgency and routing accuracy. Citation accuracy here is a label-based proxy (cited document among expected_doc_ids); it is not the human claim-by-claim review the Evaluation Framework requires, and hallucination rate is not measured. Responses are generated sentences (PR-02); the citation support rate is an automated proxy from deterministic string checks, so hallucination still needs the two-assessor human review (human_review_sample.csv). Response time is processing latency for automatic answers only; human queue time, CSAT and repeat contacts cannot be measured offline. Segments with few tickets (e.g. 8 enterprise tickets in validation) make variation figures unstable.
