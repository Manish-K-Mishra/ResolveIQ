"""Retrieval (Build Spec "Retrieve"): traceable passages, relevance floor, fallback, determinism."""
import os
import pytest
from cloudserve.generate import EvidenceSelection
from cloudserve.guardrails import validate_response
from cloudserve.ingest import load_json
from cloudserve.retrieve import (CHUNKINGS, DEFAULT_MODEL_DIR, WINDOW_CHARS, EmbeddingModelUnavailable,
                                 KnowledgeBase, OnnxSentenceEmbedder, build_passages)
from cloudserve.runtime import build_knowledge_base
from conftest import DATA, LOGIN

DOCS = load_json(DATA / "documentation.json")
CONTENT = {d["doc_id"]: d["content"] for d in DOCS}
HAVE_MODEL = (DEFAULT_MODEL_DIR / "model.onnx").exists() and (DEFAULT_MODEL_DIR / "tokenizer.json").exists()
# The only skip condition in the suite: the embedding model has not been fetched on a local
# machine. In CI (CI=true, set by GitHub Actions) these tests always run; CI fetches the model
# first, and a missing model fails test_embedding_model_is_present_in_ci instead of skipping.
IN_CI = os.getenv("CI", "").lower() == "true"
needs_model = pytest.mark.skipif(not HAVE_MODEL and not IN_CI,
                                 reason="embedding model not fetched (python scripts/fetch_models.py)")


def test_embedding_model_is_present_in_ci():
    assert HAVE_MODEL or not IN_CI, "CI must run python scripts/fetch_models.py before the tests"


@pytest.fixture(scope="module")
def embedder():
    return OnnxSentenceEmbedder()


@pytest.mark.parametrize("chunking", CHUNKINGS)
def test_every_passage_id_resolves_to_exact_corpus_text(chunking):
    kb = KnowledgeBase(DOCS, chunking=chunking)
    ids = [p["passage_id"] for p in kb.passages]
    assert len(ids) == len(set(ids)) and {p["doc_id"] for p in kb.passages} == set(CONTENT)
    for p in kb.passages:
        assert p["passage_id"].startswith(p["doc_id"] + "#")
        assert p["text"] and CONTENT[p["doc_id"]][p["start"]:p["end"]] == p["text"] == kb.resolve(p["passage_id"])
    assert kb.resolve("DOC-NOPE-001#article") is None


def test_section_and_window_chunking_follow_their_rules():
    sections = build_passages(DOCS, "section")
    assert {p["passage_id"].split("#")[1] for p in sections} == {"symptoms", "common-causes", "resolution", "notes"}
    assert all(p["text"].startswith("## ") for p in sections)
    windows = build_passages(DOCS, "window")
    assert all(len(p["text"]) <= WINDOW_CHARS for p in windows)
    by_doc = {}
    for p in windows:
        by_doc.setdefault(p["doc_id"], []).append(p)
    for ps in by_doc.values():   # consecutive windows overlap and together cover the article
        assert all(b["start"] < a["end"] for a, b in zip(ps, ps[1:]))


def test_every_article_has_a_complete_resolution_answer_unit(kb):
    for doc_id, p in kb.resolution_by_doc.items():
        assert p["passage_id"] == f"{doc_id}#resolution" and p["text"].startswith("## Resolution")
        assert kb.resolve(p["passage_id"]) == p["text"]
    assert set(kb.resolution_by_doc) == set(CONTENT)


@pytest.mark.parametrize("q", ["hyperquantum zebra orbit magnet", "", "the and for",
                               "What is the weather in Paris tomorrow?"])
@pytest.mark.parametrize("chunking", CHUNKINGS)
def test_lexical_floor_returns_nothing_for_irrelevant_queries(q, chunking):
    assert KnowledgeBase(DOCS, chunking=chunking).search(q) == []


@needs_model
@pytest.mark.parametrize("q", ["hyperquantum zebra orbit magnet", "Can you recommend a good pizza place?",
                               "What is the weather in Paris tomorrow?"])
def test_embedding_floor_returns_nothing_for_irrelevant_queries(q, embedder):
    assert KnowledgeBase(DOCS, chunking="section", retriever="embedding", embedder=embedder).search(q) == []


@needs_model
def test_embedding_search_finds_the_right_article_and_is_deterministic(embedder):
    q = "Invalid credentials when I log in to the console, my account seems locked"
    a = KnowledgeBase(DOCS, chunking="section", retriever="embedding", embedder=embedder).search(q)
    b = KnowledgeBase(DOCS, chunking="section", retriever="embedding", embedder=OnnxSentenceEmbedder()).search(q)
    assert a and a[0]["doc_id"] == "DOC-AUTH-001" and a == b
    assert all(0 < h["score"] <= 1 for h in a) and a == sorted(a, key=lambda h: (-h["score"], h["passage_id"]))


@pytest.mark.parametrize("chunking", CHUNKINGS)
def test_lexical_search_is_deterministic(chunking):
    q = "webhook deliveries failing with signature errors after rotating the secret"
    assert KnowledgeBase(DOCS, chunking=chunking).search(q) == KnowledgeBase(list(reversed(DOCS)), chunking=chunking).search(q)


def embedding_policy(policy):
    return {**policy, "policy_version": "RP-test",
            "retrieval": {"retriever": "embedding", "chunking": "section", "name": "embedding:test"},
            "retrieval_fallback": {"retriever": "lexical", "chunking": "article",
                                   "classification_confidence_min": 0.7, "retrieval_score_min": 0.4}}


def test_missing_model_falls_back_to_lexical_and_every_record_says_so(policy, nb, tmp_path, capsys):
    from cloudserve.graph import DecisionEngine
    kb, effective, info = build_knowledge_base(DOCS, embedding_policy(policy), model_dir=tmp_path)
    assert kb.retriever == "lexical" and kb.chunking == "article"
    assert info["fallback"] is True and "not found" in info["fallback_reason"]
    assert effective["retrieval_score_min"] == 0.4 and effective["classification_confidence_min"] == 0.7
    assert "lexical fallback" in capsys.readouterr().err
    r = DecisionEngine(kb, effective, nb, retrieval_info=info).process(LOGIN)
    assert r["retrieval"]["fallback"] is True and r["retrieval"]["retriever"] == "lexical"
    bad = DecisionEngine(kb, effective, nb, retrieval_info=info).process({"ticket_id": "X", "channel": "fax"})
    assert bad["retrieval"]["fallback"] is True


def test_missing_model_without_a_fallback_stops_with_a_clear_message(policy, tmp_path):
    pol = {**embedding_policy(policy), "retrieval_fallback": None}
    with pytest.raises(SystemExit, match="no fallback"):
        build_knowledge_base(DOCS, pol, model_dir=tmp_path)


def test_embedder_reports_missing_files(tmp_path):
    with pytest.raises(EmbeddingModelUnavailable, match="fetch_models"):
        OnnxSentenceEmbedder(tmp_path)


def test_record_has_passage_sources_and_a_citation_span_that_resolves(make_engine, kb):
    r = make_engine().process(LOGIN)
    assert r["action_taken"] == "auto_respond" and r["retrieval"]["fallback"] is False
    assert all(s["passage_id"] and kb.resolve(s["passage_id"]) for s in r["sources_used"])
    (span,) = r["citation_spans"]
    assert r["citations"] == [span["passage_id"]] and span["passage_id"].endswith("#resolution")
    quoted = CONTENT[span["doc_id"]][span["start"]:span["end"]]
    assert quoted and quoted in kb.resolve(span["passage_id"]) and quoted in r["response"]
    assert span["matched_passage_id"] in {s["passage_id"] for s in r["sources_used"]}


def test_quote_from_another_passage_of_the_same_article_is_blocked(kb):
    first = next(p for p in kb.passages if p["doc_id"] == "DOC-AUTH-001" and "## Symptoms" in p["text"])
    ev = kb.answer_passages([{"passage_id": first["passage_id"], "doc_id": "DOC-AUTH-001", "score": 0.9}])
    symptoms = first["text"][first["text"].index("## Symptoms"):][:200]
    assert symptoms not in ev[0]["passage"]
    v = validate_response(EvidenceSelection("DOC-AUTH-001", symptoms, "DOC-AUTH-001#resolution"), ev)
    assert not v["passed"] and "quote_not_supported_by_source" in v["failures"]
    v = validate_response(EvidenceSelection("DOC-AUTH-001", ev[0]["passage"][20:200], "DOC-BILL-001#resolution"), ev)
    assert not v["passed"] and "invalid_document_id" in v["failures"]
