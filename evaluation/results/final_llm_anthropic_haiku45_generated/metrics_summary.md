# Evaluation summary — run-20261121T195724-7d4837

Input `data/validation_tickets.json` (80 tickets), engine `langgraph`, classifier `anthropic:claude-haiku-4-5 (PR-01 v1.1)`, answer mode `generated`, policy `RP-1.0`, automatic sending OFF (kill switch), 2026-11-21T20:01:18+00:00.

| Measure | Value |
|---|---|
| Processed / logged | 80 / 80 (reconciles: True) |
| Auto-responded (sent) / held by kill switch | 0 / 19 |
| Escalated / blocked | 51 / 10 |
| FCR (actual / if released) | 0.0% / 23.8% |
| Escalation rate (actual / if released) | 100.0% / 76.2% |
| Latency median / p95 (s), live decisions only | 1.8897 / 5.1164 |
| Private data in outbound responses | 0 |
| Model calls (live / cache hits) | 109 / 0 (cache mode: read_write) |
| Tokens in / out; estimated cost this run / if all live (USD) | 161844 / 20666; 0.265174 / 0.265174 |
| Projected monthly cost (USD, 2167 tickets/month) | 7.1829 |
| Retriever (fallback) | lexical:term-overlap, article passages (fallback: False) |
| Cited quotes found at their recorded corpus span | 1.0 |
| Generated answers (released or held) / cannot_answer / generation failed | 19 / 0 / 0 |
| Blocked by grounding checks / unsupported claims flagged | 10 / 11 |
| Citations per answer / citation support rate (automated proxy) | 1.895 / 0.9206 |
| Intent accuracy / macro precision / macro recall | 0.9125 / 0.9286 / 0.9091 |
| Retrieval hit rate (any / top-1) | 0.8113 / 0.7358 |
| Citation accuracy vs expected docs (proxy) | 0.9412 |
| Auto precision / recall vs expected route | 0.8947 / 0.3542 |
| must_not_auto_respond violations | 0 of 14 |
| Calibration max gap (points) | 55.0 |

**Limitations.** Training overlap: 62 of 80 ticket bodies in this input are exact copies of development tickets the offline classifier was trained on; intent accuracy on those measures recall of training data, not generalisation, so treat it with caution (docs/architecture.md §8). Urgency and expected_route labels vary for identical ticket text in the supplied data, which caps achievable urgency and routing accuracy. Citation accuracy here is a label-based proxy (cited document among expected_doc_ids); it is not the human claim-by-claim review the Evaluation Framework requires, and hallucination rate is not measured. Responses are generated sentences (PR-02); the citation support rate is an automated proxy from deterministic string checks, so hallucination still needs the two-assessor human review (human_review_sample.csv). Response time is processing latency for automatic answers only; human queue time, CSAT and repeat contacts cannot be measured offline. Segments with few tickets (e.g. 8 enterprise tickets in validation) make variation figures unstable.
