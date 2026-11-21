"""Offline classifier (A3): taxonomy, probability, alternatives, no label leakage."""
from cloudserve.classify import INTENTS, UNKNOWN, NaiveBayesIntentClassifier
from cloudserve.ingest import normalize
from conftest import LOGIN


def test_output_is_in_the_22_intent_taxonomy_with_probability(nb):
    c = nb.classify(normalize(LOGIN))
    assert len(INTENTS) == 22
    assert c["intent"] in INTENTS
    assert 0.0 <= c["confidence"] <= 1.0
    assert c["urgency"] in ("low", "medium", "high")
    assert 1 <= len(c["alternatives"]) <= 3
    assert all(a["intent"] in INTENTS and a["intent"] != c["intent"] for a in c["alternatives"])


def test_posterior_sums_to_one(nb):
    assert abs(sum(p for _, p in nb.posterior(LOGIN["body"])) - 1.0) < 1e-9


def test_out_of_vocabulary_ticket_returns_unknown_fallback(nb):
    c = nb.classify(normalize({"ticket_id": "Z", "channel": "forum", "body": "qwxz blorptastic zzyzx"}))
    assert c["intent"] == UNKNOWN and c["confidence"] == 0.0 and c["alternatives"] == []


def test_classifier_is_deterministic(nb):
    t = normalize(LOGIN)
    assert nb.classify(t) == nb.classify(t)


def test_training_ignores_labels_outside_taxonomy_and_versions_the_training_data():
    a = NaiveBayesIntentClassifier([("login fails", "authentication_failure"), ("odd", "not_a_label")])
    b = NaiveBayesIntentClassifier([("login fails", "authentication_failure")])
    assert set(a.class_counts) == {"authentication_failure"}
    assert a.version != b.version and a.version.startswith("nb-")


def test_prediction_does_not_change_when_labels_are_supplied(nb):
    with_labels = normalize({**LOGIN, "labels": {"intent": "billing_query"}})
    assert nb.classify(with_labels) == nb.classify(normalize(LOGIN))
