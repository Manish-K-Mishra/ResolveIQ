"""Guardrails (FR-06, A6, A7, Governance §4): the checks that can block or escalate a response.

Patterns screen untrusted ticket text (injection, sensitive matters) and every outbound response
(grounding, PII, instruction integrity, scope) before release. Deterministic, no model calls.
"""
from __future__ import annotations
import re
from .generate import EvidenceSelection, render_response

# Defence in depth; intent-based routing is the primary control.
PII_PATTERN = re.compile(
    r"(?i)(?:\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b"            # email address
    r"|\b(?:\+?\d[\d ()-]{8,}\d)\b"                       # phone-like number
    r"|\b(?:sk|pk|rk)[-_](?:live|test)[-_][a-z0-9]{8,}\b"  # API-key shapes
    r"|\bAKIA[0-9A-Z]{16}\b"                              # cloud access key id
    r"|\b(?:\d[ -]?){13,16}\b)")                          # card-number-like digit run
INJECTION_PATTERN = re.compile(
    r"(?i)(?:\b(?:ignore|disregard|forget|override)\s+(?:(?:all|any|the|your|my|these|those|previous|prior|above|earlier|system|of)\s+)*"
    r"(?:instructions?|rules?|prompts?|guidelines?|directions?|directives?)\b"
    r"|\bforget\s+(?:everything|all)\s+(?:you\s+)?(?:were|have\s+been|was)\s+told"
    r"|\b(?:reveal|print|show|tell|display|repeat|output|leak)\s+(?:me\s+)?(?:the\s+|your\s+)?(?:hidden\s+|secret\s+|initial\s+|original\s+)?"
    r"(?:system\s+prompt|instructions|api\s+key)"
    r"|\bbypass\s+(?:the\s+|your\s+)?(?:guardrails|filters|safety)"
    r"|\byou\s+are\s+now\s+(?:a|an|the|dan|in)\b"
    r"|\b(?:unrestricted|unfiltered|jailbroken)\s+(?:ai|assistant|mode|model)\b"
    r"|\bact\s+as\s+(?:an?\s+|the\s+)?(?:admin|administrator|developer|system|root|superuser|dan)\b"
    r"|\bnew\s+(?:system\s+)?instructions?\s*:"
    # Paraphrased redirection: replacing the system's procedures or asking for its internals.
    r"|\b(?:follow|obey|use|adopt|apply)\s+(?:these|this|my|the\s+following)\s+(?:new\s+|updated\s+|revised\s+)?"
    r"(?:operating\s+)?(?:procedures?|protocols?|policies|policy|orders|commands)\b"
    # The qualifier is required ("your", "internal", "hidden", "system"), so "show me the
    # configuration of my SSO connection" or "the policy for refunds" is not flagged.
    r"|\b(?:reveal|print|show|display|repeat|output|leak|disclose|dump)\s+(?:me\s+)?(?:the\s+)?"
    r"(?:your\s+(?:full\s+|internal\s+|hidden\s+|support\s+|own\s+)?|(?:full\s+)?(?:internal|hidden|system)\s+(?:support\s+)?)"
    r"(?:prompt|policy|policies|configuration|guidelines|rules)\b"
    r"|\bfrom\s+now\s+on\s+(?:you|your)\b"
    r"|<\s*/?\s*system\s*>"
    r"|\bignora\s+(?:todas\s+)?(?:las\s+)?instrucciones)")
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "@": "a", "$": "s"})


def contains_injection(text: str) -> bool:
    """Instruction-integrity screen on untrusted ticket text (Governance §4).

    Checks the text as written and a normalised copy (NFKC, zero-width characters removed,
    common digit-for-letter swaps undone) so trivial obfuscation does not slip past. This is
    still pattern matching: paraphrased injections can pass (docs/architecture.md §10).
    The answer released to the customer is a verified documentation excerpt either way.
    """
    import unicodedata
    plain = unicodedata.normalize("NFKC", text or "").translate(_ZERO_WIDTH)
    return bool(INJECTION_PATTERN.search(plain) or INJECTION_PATTERN.search(plain.translate(_LEET)))


RISK_PATTERN = re.compile(
    r"(?i)\b(?:legal\s+action|lawsuit|lawyers?|attorneys?|sue|suing|subpoena"
    r"|breach(?:ed|es)?|data\s+leak|leaked|compromised"
    r"|credentials?\s+(?:may\s+have\s+been\s+|have\s+been\s+|were\s+|was\s+)?exposed"
    r"|exposed\s+(?:credentials?|keys?|secrets?|data)"
    r"|refund(?:ed|s)?|chargeback)\b")


# Pre-release response checks.
DOC_ID = re.compile(r"^DOC-[A-Z0-9-]+$")
PASSAGE_ID = re.compile(r"^DOC-[A-Z0-9-]+#[a-z0-9-]+$")
# Tone/scope: the system must not claim completed account actions or make commitments.
SCOPE_PATTERN = re.compile(
    r"(?i)\b(?:we\s+have\s+(?:issued|refunded|credited|fixed|rotated|changed|reset|restored)"
    r"|we\s+(?:will|guarantee\s+to)\s+(?:refund|credit|compensate)|guaranteed?)\b")


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def mask_pii(text: str) -> str:
    return PII_PATTERN.sub("[redacted]", text)


def validate_response(selection: EvidenceSelection | None, evidence: list[dict]) -> dict:
    """Pre-release checks. Returns {'passed', 'failures', 'guardrail_results', 'response'}.

    grounding: the cited passage (or, without a passage id, the cited doc) must be one of the
    approved evidence passages and the quote must occur verbatim in that passage's text. pii: no private-data patterns in the outbound text. instruction_integrity:
    no instruction-like text in the outbound text. scope: no completed-action claims/commitments.
    """
    results = {"grounding": "pass", "pii": "pass", "instruction_integrity": "pass", "scope": "pass"}
    failures: list[str] = []
    if selection is None:
        results = {k: "not_run" for k in results}
        results["grounding"] = "fail"
        return {"passed": False, "failures": ["no_evidence_selection"], "guardrail_results": results, "response": ""}
    if selection.passage_id is None:
        source = next((m["passage"] for m in evidence if m["doc_id"] == selection.doc_id), None)
    else:
        source = next((m["passage"] for m in evidence if m.get("passage_id") == selection.passage_id
                       and m["doc_id"] == selection.doc_id), None)
    if not DOC_ID.fullmatch(selection.doc_id) or (
            selection.passage_id is not None and not (PASSAGE_ID.fullmatch(selection.passage_id)
                                                      and selection.passage_id.startswith(selection.doc_id + "#"))):
        failures.append("invalid_document_id")
    if source is None:
        failures.append("citation_not_in_retrieval")
    if not selection.quote or len(selection.quote.strip()) < 40 or len(selection.quote) > 1500:
        failures.append("quote_invalid_length")
    elif source is not None and _normalize(selection.quote) not in _normalize(source):
        failures.append("quote_not_supported_by_source")
    if failures:
        results["grounding"] = "fail"
    response = render_response(selection)
    if PII_PATTERN.search(response):
        failures.append("possible_private_data"); results["pii"] = "fail"
    if INJECTION_PATTERN.search(response):
        failures.append("suspicious_source_instructions"); results["instruction_integrity"] = "fail"
    if SCOPE_PATTERN.search(response):
        failures.append("out_of_scope_commitment"); results["scope"] = "fail"
    passed = not failures
    return {"passed": passed, "failures": sorted(set(failures)), "guardrail_results": results,
            "response": response if passed else ""}


# ---------------- generated answers (PR-02) ----------------
# Commitments a generated answer must never make (Governance §4 tone and scope): refunds,
# credits, compensation, timelines, or claims that an account action was already taken.
GENERATED_SCOPE_PATTERN = re.compile(
    r"(?i)\b(?:(?:we|i)\s*(?:will|'ll|shall|are\s+going\s+to)\s+(?:\w+\s+){0,2}(?:refund|credit|compensate|reimburse|waive)"
    r"|(?:refund|credit|reimbursement|compensation)s?\s+(?:has|have|will)\s+been?\s*(?:be\s+)?(?:issued|applied|processed|approved|granted)"
    r"|(?:we|i)\s*(?:will|'ll|shall)\b[^.\n]{0,80}\b(?:within|by)\s+(?:\d+|one|two|three|24|48|the\s+end|tomorrow|today|next)"
    r"|(?:we|i)\s+(?:have|'ve)\s+(?:already\s+)?(?:issued|refunded|credited|fixed|rotated|changed|reset|restored|unlocked|resolved)"
    r"|guarantee[ds]?)\b")
# Strict terms: each must appear in the cited passages. Numbers, inline code / CLI commands,
# URLs, flags, paths, dotted or snake_case keys, ALL-CAPS identifiers and words with digits.
_STRICT_TERMS = [
    re.compile(r"`([^`]+)`"),                                     # inline code, CLI commands
    re.compile(r"\bhttps?://[^\s)\]]+"),                          # URLs
    re.compile(r"(?<![\w-])--?[a-z][\w-]*"),                      # CLI flags
    re.compile(r"\b[\w-]+(?:/[\w.-]+)+"),                         # paths
    re.compile(r"\b[a-z][a-z0-9]*(?:[._][a-z0-9]+)+\b"),          # dotted / snake_case keys
    re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b"),                        # CONSTANTS, codes
    re.compile(r"\b\d+(?:[.,:]\d+)*%?"),                          # numbers
]
_ALLOWED_UPPER = {"API", "CLI", "SSO", "URL", "SDK", "UTC", "CSV", "JSON", "HTTP", "HTTPS", "DNS", "TLS", "SAML", "OIDC"}
# Words a faithful answer adds around documentation facts; they carry no claim of their own.
_FRAME_WORDS = {"cloudserve", "documentation", "doc", "docs", "steps", "step", "follow", "following",
                "recommend", "recommended", "suggest", "try", "able", "make", "sure", "ensure", "check",
                "first", "next", "finally", "also", "may", "might", "could", "customer", "account", "issue",
                "problem", "resolve", "resolving", "resolution", "case", "here", "these", "those", "below",
                "guide", "article", "section", "note", "notes", "help"}
SUPPORT_MIN_OVERLAP = 0.6   # share of a sentence's content words found in its cited passages


def _stem(w: str) -> str:
    """Crude, deterministic stemming so rotate / rotates / rotating / rotation compare equal."""
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "es", "ed", "ly", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)][:6]
    return w[:6]


def strict_terms(text: str) -> list[str]:
    found = []
    for pat in _STRICT_TERMS:
        for m in pat.finditer(text):
            term = (m.group(1) if pat.groups else m.group(0)).strip().rstrip(".,;:")
            if term and term not in _ALLOWED_UPPER and term not in found:
                found.append(term)
    return found


def support_check(sentence: str, passages_text: str) -> dict:
    """Deterministic support of one sentence by the text of the passages it cites.

    strict: every strict term (number, command, URL, flag, path, key, identifier) occurs in the
    passages (case-insensitive substring). overlap: at least SUPPORT_MIN_OVERLAP of the
    sentence's content words (stop and frame words removed, crude stems) occur in the passages.
    """
    from .ingest import tokens
    low = passages_text.lower()
    missing_strict = [t for t in strict_terms(sentence) if t.lower() not in low]
    words = {_stem(w) for w in tokens(sentence) if w not in _FRAME_WORDS and not w.isdigit()}
    have = {_stem(w) for w in tokens(passages_text)}
    missing_words = sorted(words - have)
    overlap = 1.0 if not words else round(len(words & have) / len(words), 4)
    return {"strict_ok": not missing_strict, "overlap": overlap,
            "overlap_ok": overlap >= SUPPORT_MIN_OVERLAP,
            "missing_terms": missing_strict + missing_words}


def validate_generated(answer, approved: list[dict]) -> dict:
    """Pre-release checks on a generated answer. Runs on every generated answer, blocked or not.

    Returns {'passed', 'cannot_answer', 'failures', 'guardrail_results', 'response',
    'sentences' (per-sentence checks), 'unsupported_claims', 'citation_spans'}.
    """
    from .generate import render_generated
    by_id = {p["passage_id"]: p for p in approved}
    results = {"citations": "pass", "grounding": "pass", "support": "pass",
               "pii": "pass", "instruction_integrity": "pass", "scope": "pass"}
    if answer.cannot_answer:
        results = {k: "not_run" for k in results}
        return {"passed": False, "cannot_answer": True, "failures": ["documentation_does_not_answer"],
                "guardrail_results": results, "response": "", "sentences": [], "unsupported_claims": [],
                "citation_spans": [], "reason": answer.reason}
    failures, checked, unsupported, spans = [], [], [], []
    if not answer.sentences:
        failures.append("empty_answer")
        results["grounding"] = "fail"
    for i, s in enumerate(answer.sentences):
        cites = list(s.get("citations") or [])
        known = [c for c in cites if c in by_id]
        problems = []
        if not cites:
            problems.append("no_citation")
        if len(known) != len(cites):
            problems.append("citation_not_in_approved_passages")
        # Verbatim after whitespace normalisation; surrounding quote marks and a trailing full stop
        # or comma (models end a copied list item with one) are not part of the span.
        quote = _normalize(s.get("quote") or "").strip(" \"'“”").rstrip(".,;:").rstrip()
        home = next((c for c in known if quote and quote in _normalize(by_id[c]["passage"])), None)
        if len(quote) < 15 or len(quote.split()) < 3:
            problems.append("quote_too_short")
        elif home is None:
            problems.append("quote_not_verbatim_in_cited_passage")
        cited_text = "\n".join(by_id[c]["passage"] for c in known)
        sup = support_check(s["text"], cited_text) if known else {"strict_ok": False, "overlap": 0.0,
                                                                   "overlap_ok": False, "missing_terms": []}
        if not sup["strict_ok"]:
            problems.append("unsupported_specifics")
        if not sup["overlap_ok"]:
            problems.append("low_overlap_with_cited_passages")
        # Per citation: does this passage alone support the sentence (automated citation proxy)?
        per_citation = {c: (c in by_id and all(support_check(s["text"], by_id[c]["passage"])[k]
                                               for k in ("strict_ok", "overlap_ok"))) for c in cites}
        entry = {"index": i, "text": s["text"], "citations": cites, "quote": s.get("quote", ""),
                 "supported": not problems, "problems": problems, "overlap": sup["overlap"],
                 "missing_terms": sup["missing_terms"], "citation_support": per_citation}
        checked.append(entry)
        if problems:
            unsupported.append({"sentence": s["text"], "citations": cites, "problems": problems,
                                "missing_terms": sup["missing_terms"]})
        elif home is not None:
            p = by_id[home]
            at = _normalize(p["passage"]).find(quote)
            raw_at = p["passage"].find(quote)
            spans.append({"sentence": i, "passage_id": home, "doc_id": p["doc_id"],
                          "start": p["start"] + raw_at if raw_at >= 0 else None,
                          "end": p["start"] + raw_at + len(quote) if raw_at >= 0 else None,
                          "quote_found_normalised": at >= 0})
    if any("citation_not_in_approved_passages" in e["problems"] or "no_citation" in e["problems"] for e in checked):
        results["citations"] = "fail"
        failures.append("invalid_citation")
    if any(p in e["problems"] for e in checked for p in ("quote_too_short", "quote_not_verbatim_in_cited_passage")):
        results["grounding"] = "fail"
        failures.append("quote_not_supported_by_source")
    if any(p in e["problems"] for e in checked for p in ("unsupported_specifics", "low_overlap_with_cited_passages")):
        results["support"] = "fail"
        failures.append("unsupported_claim")
    response = render_generated(answer, approved) if answer.sentences else ""
    if PII_PATTERN.search(response):
        failures.append("possible_private_data"); results["pii"] = "fail"
    if contains_injection(response):
        failures.append("suspicious_source_instructions"); results["instruction_integrity"] = "fail"
    if SCOPE_PATTERN.search(response) or GENERATED_SCOPE_PATTERN.search(response):
        failures.append("out_of_scope_commitment"); results["scope"] = "fail"
    passed = not failures
    return {"passed": passed, "cannot_answer": False, "failures": sorted(set(failures)),
            "guardrail_results": results, "response": response if passed else "", "sentences": checked,
            "unsupported_claims": unsupported, "citation_spans": spans if passed else [], "reason": answer.reason}
