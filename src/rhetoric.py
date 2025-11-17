from __future__ import annotations

from typing import List
import re


# Very simple keyword-based heuristics for rhetorical interpretation.

SELF_PRONOUNS = re.compile(r"\b(I|I'm|I’m|me|my|mine|we|we're|we’re|our|ours)\b", re.IGNORECASE)

OPPONENT_WORDS = [
    "clinton", "hillary", "crooked hillary",
    "biden", "joe biden",
    "trump", "donald trump",
    "republicans", "democrats", "the left", "the right",
    "gop"
]
OPPONENT_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in OPPONENT_WORDS) + r")\b", re.IGNORECASE)

TOPIC_KEYWORDS = {
    "economy": [
        "economy", "jobs", "job", "unemployment", "wages", "salary",
        "tax", "taxes", "trade", "tariffs", "inflation", "growth", "market"
    ],
    "healthcare": [
        "health care", "healthcare", "obamacare", "affordable care",
        "medicare", "medicaid", "insurance", "patients", "doctors", "covid", "pandemic", "virus"
    ],
    "immigration": [
        "immigration", "immigrants", "migrants", "border", "wall", "refugees",
        "asylum", "deport", "visa", "daca"
    ],
    "security": [
        "crime", "criminals", "law and order", "police", "terror", "terrorism",
        "isis", "security", "dangerous", "threat"
    ],
    "foreign_policy": [
        "russia", "china", "iran", "north korea", "allies", "nato", "war",
        "troops", "military", "foreign policy", "diplomacy"
    ],
    "environment": [
        "climate", "climate change", "global warming", "environment",
        "pollution", "clean energy", "renewable", "paris agreement"
    ],
    "rights_justice": [
        "rights", "civil rights", "voting rights", "equality", "justice",
        "discrimination", "racism", "race", "gender", "lgbt", "marriage equality"
    ],
}


def tag_target_type(text: str) -> str:
    """Classify the *target* of the message in a very rough way.

    Returns one of: 'self', 'opponent', 'issue', 'other'.
    """
    if not isinstance(text, str) or not text.strip():
        return "other"

    t = text.lower()

    if OPPONENT_RE.search(t):
        return "opponent"

    if SELF_PRONOUNS.search(t):
        return "self"

    for _, words in TOPIC_KEYWORDS.items():
        for w in words:
            if w in t:
                return "issue"

    return "other"


def tag_topic(text: str) -> str:
    """Assign a coarse topic label based on keyword matches.

    Returns one of the keys in TOPIC_KEYWORDS, or 'other' if nothing matches.
    """
    if not isinstance(text, str) or not text.strip():
        return "other"

    t = text.lower()

    for topic, words in TOPIC_KEYWORDS.items():
        for w in words:
            if w in t:
                return topic

    return "other"
