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

    grounding: cited doc must be one of the approved retrieved articles and the quote must occur
    verbatim in it. pii: no private-data patterns in the outbound text. instruction_integrity:
    no instruction-like text in the outbound text. scope: no completed-action claims/commitments.
    """
    results = {"grounding": "pass", "pii": "pass", "instruction_integrity": "pass", "scope": "pass"}
    failures: list[str] = []
    if selection is None:
        results = {k: "not_run" for k in results}
        results["grounding"] = "fail"
        return {"passed": False, "failures": ["no_evidence_selection"], "guardrail_results": results, "response": ""}
    source = next((m["passage"] for m in evidence if m["doc_id"] == selection.doc_id), None)
    if not DOC_ID.fullmatch(selection.doc_id):
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
