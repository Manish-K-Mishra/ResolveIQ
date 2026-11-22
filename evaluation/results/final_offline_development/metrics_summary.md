# Evaluation summary — run-20261122T040344-a30285

Input `data/development_tickets.json` (500 tickets), engine `langgraph`, classifier `naive_bayes_intent (nb-a8a09fec6990)`, answer mode `extractive`, policy `RP-1.0`, automatic sending OFF (kill switch), 2026-11-22T04:03:45+00:00.

| Measure | Value |
|---|---|
| Processed / logged | 500 / 500 (reconciles: True) |
| Auto-responded (sent) / held for review (auto-send off; runtime kill switch) | 0 / 210 (210; 0) |
| Escalated / blocked | 290 / 0 |
| Response time to decision, mean / median (s), all tickets | 0.0012 / 0.0011 |
| FCR (actual / if released) | 0.0% / 42.0% |
| Escalation rate (actual / if released) | 100.0% / 58.0% |
| Latency median / p95 (s), live decisions only | 0.0011 / 0.0014 |
| Private data in outbound responses | 0 |
| Model calls (live / cache hits) | 0 / 0 (cache mode: not_used (no model in this run)) |
| Tokens in / out; estimated cost this run / if all live (USD) | 0 / 0; 0 / 0 |
| Projected monthly cost (USD, 2167 tickets/month) | 0.0 |
| Retriever (fallback) | lexical:term-overlap, article passages (fallback: False) |
| Cited quotes found at their recorded corpus span | 1.0 |
| Intent accuracy / macro precision / macro recall | 0.998 / 0.9982 / 0.997 |
| Retrieval hit rate (any / top-1) | 0.7955 / 0.7395 |
| Citation accuracy vs expected docs (proxy) | 0.9029 |
| Auto precision / recall vs expected route | 0.8095 / 0.5466 |
| must_not_auto_respond violations | 0 of 87 |
| Calibration max gap (points) | 21.9 |

**Limitations.** Training overlap: 500 of 500 ticket bodies in this input are exact copies of development tickets the offline classifier was trained on; intent accuracy on those measures recall of training data, not generalisation, so treat it with caution (docs/architecture.md §8). Urgency and expected_route labels vary for identical ticket text in the supplied data, which caps achievable urgency and routing accuracy. Citation accuracy here is a label-based proxy (cited document among expected_doc_ids); it is not the human claim-by-claim review the Evaluation Framework requires, and hallucination rate is not measured. Responses are extractive (a verbatim documentation section), not generated text. Response time is system processing time (time to decision; for automatic answers also time to first reply); human queue time, CSAT and repeat contacts cannot be measured offline. Segments with few tickets (e.g. 8 enterprise tickets in validation) make variation figures unstable.
