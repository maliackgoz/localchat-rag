"""Query router and filtered similarity search. Contract: agents/04_retrieval.md."""

from retrieval.retriever import DEFAULT_MIN_SIM, RetrievalResult, Retriever
from retrieval.router import Roster, classify_intent, load_roster

__all__ = [
    "DEFAULT_MIN_SIM",
    "RetrievalResult",
    "Retriever",
    "Roster",
    "classify_intent",
    "load_roster",
]
