from pathlib import Path
import pytest
from cloudserve.classify import NaiveBayesIntentClassifier
from cloudserve.graph import DecisionEngine
from cloudserve.ingest import load_json
from cloudserve.retrieve import KnowledgeBase
from cloudserve.route import load_policy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="session")
def kb():
    return KnowledgeBase.from_path(DATA / "documentation.json")


@pytest.fixture(scope="session")
def policy():
    return load_policy()


@pytest.fixture(scope="session")
def nb():
    return NaiveBayesIntentClassifier.from_development_file(DATA / "development_tickets.json")


@pytest.fixture(scope="session")
def validation():
    return load_json(DATA / "validation_tickets.json")


@pytest.fixture
def make_engine(kb, policy, nb):
    def _make(classifier=None, selector=None, auto_send=False, use_langgraph=True):
        return DecisionEngine(kb, policy, classifier or nb, selector, auto_send=auto_send, use_langgraph=use_langgraph)
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
