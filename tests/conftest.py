from pathlib import Path
import pytest
from cloudserve.classify import NaiveBayesIntentClassifier
from cloudserve.graph import DecisionEngine
from cloudserve.ingest import load_json
from cloudserve.route import load_policy
from cloudserve.runtime import build_knowledge_base

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="session")
def retrieval():
    """The retriever the committed policy was derived for, built exactly as the runtime does."""
    return build_knowledge_base(load_json(DATA / "documentation.json"), load_policy())


@pytest.fixture(scope="session")
def kb(retrieval):
    return retrieval[0]


@pytest.fixture(scope="session")
def policy(retrieval):
    return retrieval[1]


@pytest.fixture(scope="session")
def nb():
    return NaiveBayesIntentClassifier.from_development_file(DATA / "development_tickets.json")


@pytest.fixture(scope="session")
def validation():
    return load_json(DATA / "validation_tickets.json")


@pytest.fixture
def make_engine(retrieval, nb):
    kb, policy, info = retrieval

    def _make(classifier=None, selector=None, auto_send=False, use_langgraph=True):
        return DecisionEngine(kb, policy, classifier or nb, selector, auto_send=auto_send,
                              use_langgraph=use_langgraph, retrieval_info=info)
    return _make


class FixedClassifier:
    """Test double: returns a fixed classification (lets routing be tested in isolation)."""
    name, version = "fixed", "test"

    def __init__(self, intent="authentication_failure", confidence=0.99, urgency="medium"):
        self.out = {"intent": intent, "confidence": confidence, "urgency": urgency, "alternatives": [],
                    "reason": "test", "confidence_type": "test_fixed"}

    def classify(self, ticket):
        return dict(self.out)


# An answerable, low-risk ticket in the style of the corpus (DOC-AUTH-001).
LOGIN = {"ticket_id": "T-LOGIN", "channel": "email", "subject": "Invalid credentials login",
         "body": "Account locked with invalid credentials in console. How do I reset session?",
         "customer_tier": "standard"}
