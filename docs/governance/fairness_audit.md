# Fairness audit (Governance Framework §3)

**What was measured.** Every figure below comes from `scripts/fairness_audit.py`, run on the
final harness outputs. The full tables, with Wilson 95% intervals and the extra diagnostic
columns, are in `evaluation/results/fairness/<run>.md` and `.json`. Regenerate them with:

```bash
python scripts/fairness_audit.py --run evaluation/results/final_offline_default \
  --tickets data/validation_tickets.json --out evaluation/results/fairness/final_offline_default
```

**Definitions** (the same for every run):

| Term | Definition |
|---|---|
| Resolution rate | `auto_respond` decisions divided by tickets: first contact resolution if the kill switch were released. |
| Quality score | Routing accuracy: the share of tickets where the decision (answer or escalate) matches the `expected_route` label. For generated runs, the automated citation-support rate is shown as a second quality proxy. **Human quality scores are pending.** They need the two-assessor review of `human_review_sample.csv`, which has not been filled in. |
| Short / long | Short means a body of at most the median word count of the input file: 24 words for validation, 25 for development. Long means above it. |
| Small business | The dataset has three tiers (`standard`, `business`, `enterprise`) and no "small business" value. We report all three tiers. Read `standard` and `business` as the non-enterprise tiers. |
| Variation from best | The best segment's value minus this segment's value, in percentage points. The target is under 5 points. |

## Headline run: validation (80), offline, RP-1.0 (`final_offline_default`)

| Segment | Tickets | Resolution rate | Quality score (routing accuracy) | Variation from best (resolution / quality, pts) | Explanation |
|---|---|---|---|---|---|
| Enterprise | 8 | 50.0% | 75.0% | 0.0 / 0.0 | A very small group (95% CI for resolution 22–78%). Tier is not a routing input (`test_enterprise_tier_is_not_escalated_by_tier_alone`). |
| Business | 30 | 46.7% | 63.3% | 3.3 / 11.7 | 70% of these tickets are labelled `auto_respond`. 8 of the missed ones had no approved document above the floor. |
| Standard | 42 | 35.7% | 61.9% | 14.3 / 13.1 | Intent mix alone predicts 30.8%. Only 50% of standard tickets are labelled answerable automatically. |
| Fluent English | 61 | 44.3% | 65.6% | 0.0 / 0.0 | — |
| Non-fluent English | 19 | 31.6% | 57.9% | 12.7 / 7.7 | The label ceiling is 42.1% against 65.6% for fluent tickets: fewer of these tickets are answerable at all. Top-1 retrieval hit rate on this small set is 88.9% (n=9). |
| Short tickets (≤ 24 words) | 43 | 58.1% | 74.4% | 0.0 / 0.0 | — |
| Long tickets (> 24 words) | 37 | 21.6% | 51.3% | **36.5** / 23.1 | **The largest resolution gap.** Same label ceiling as short tickets (59.5% against 60.5%). Explained by retrieval; see the investigation below. |
| Channel | chat 22, docs_comment 16, email 31, forum 11 | 45.5% / 37.5% / 45.2% / 27.3% | 86.4% / 62.5% / 51.6% / 54.5% | 18.2 / 34.8 (spread) | Intent mix alone predicts 20.9 points of the resolution spread. |
| Region | AP 21, EU 25, LatAm 7, NA 27 | 42.9% / 44.0% / 28.6% / 40.7% | 61.9% / 48.0% / 85.7% / 74.1% | 15.4 / 37.7 (spread) | Region is not an input. Intent mix alone predicts 28.9 points. Latin America has 7 tickets. |

**The 5-point target is not met for any segment family** in this run, on either measure. The
segments are small, so most intervals overlap. For example, non-fluent resolution is 15–54%
against 33–57% for fluent tickets.

## Investigation of the largest gap: long tickets

The largest resolution gap is ticket length: 36.5 points on validation. (Region quality has a
37.7-point spread, but on groups of 7 to 27 tickets with intent mix predicting 28.9 points.)

1. **It is not who asks or what they ask.** Long and short tickets are equally often labelled
   answerable (59.5% against 60.5%). Intent accuracy is 100% on long tickets.
2. **It is the retrieval score.** Of the 22 long tickets labelled `auto_respond`, the system
   answered 6. Of the 16 it did not answer, 13 failed on retrieval: 8 had no approved document
   above the floor and 5 had no relevant document. The other 3 were sensitive requests. On short
   tickets, all 6 misses were `no_matching_documentation_for_intent`. The median top retrieval
   score is 0.353 for long tickets and 0.444 for short ones, against a floor of 0.40.
3. **The cause.** RP-1.0's lexical score is the share of the ticket's terms found in the passage
   (`retrieve.py`). A longer ticket has more terms, so the same correct article scores lower and
   falls under the 0.40 floor. This is a property of the score's normalisation, not of the
   customer.
4. **It holds on 500 tickets.** On the development set (`final_offline_development`; the offline
   classifier is in-sample there, but retrieval and routing are not trained), long tickets have
   28 `no_relevant_documentation` misses against 9 for short ones. Their resolution rate is 37.6%
   against 45.8%, although they are more often labelled answerable (65.5% against 59.4%).
5. **Fluency.** On development data, non-fluent tickets have a top-1 retrieval hit rate of 67.8%
   against 75.9% for fluent ones. That is the retrieval-and-phrasing effect the Governance
   Framework predicts. The 19 non-fluent validation tickets are too few to show it.

**What it would take to fix.** One option is a length-robust score, for example matching on the
passage's terms rather than the ticket's, or BM25. Another is the embedding retriever, which
had better hit rates in Slice 6 but failed the precision objective (docs/architecture.md §11).
Either change alters routing, so it needs a new derived policy version and the owner's approval.
**Not changed in this slice.**

## The same audit on the LLM runs (validation 80, generated answers)

| Run | Tier | Fluency | Length | Channel | Region | Notes |
|---|---|---|---|---|---|---|
| Ollama gemma4:e2b (`final_llm_ollama_gemma4_generated`) | 11.9 / 7.1 | 20.6 / 5.4 | 23.7 / 9.5 | 26.4 / 15.2 | 17.7 / 27.4 | Spread in points, resolution / quality. Non-fluent resolution 10.5% against 31.1%; citation support 87.5% against 98.2%. |
| Claude Haiku 4.5 (`final_llm_anthropic_haiku45_generated`) | 27.5 / 25.8 | 17.3 / 12.7 | 29.1 / 18.8 | 30.1 / 32.1 | 8.6 / 42.9 | Enterprise resolution 12.5% (1 of 8). Non-fluent citation support is 66.7% against 96.3% for fluent tickets. Long-ticket resolution is 8.1% against 37.2%. |

The automated citation-support figures for non-fluent tickets rest on very few answers (2 for
each LLM run). They point the same way as the retrieval finding, but they are not evidence on
their own.

## Development set, for scale (500, IN-SAMPLE for the classifier)

Variation in points (resolution / quality): tier 1.3 / 12.1; fluency 3.7 / 6.1; length
8.2 / 12.2; channel 13.1 / 10.5; region 18.4 / 15.3. With six times the tickets, the tier and
fluency resolution gaps fall under 5 points. The length and region gaps do not. These figures
are **not** held-out results.

## Conclusion

- **Target.** The system does not meet the under-5-point target on the 80-ticket gate. The most
  defensible finding is a **length bias from the lexical retrieval score**, confirmed on 500
  tickets. A further retrieval gap for non-fluent English shows on the development set.
- **Effect on customers.** Both gaps make the system escalate more. They do not make it answer
  worse: auto precision for long tickets is 75%, and there are no `must_not_auto_respond`
  violations in any segment. The harm is less automation for those customers, not wrong answers
  sent to them.
- **Owner decision.** Whether this is acceptable for a pilot is to be confirmed by the project owner.
