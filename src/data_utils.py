from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import TRUMP, OBAMA, BIDEN

URL_RE = re.compile(r"http\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """Basic tweet cleaning: remove URLs, collapse whitespace, normalize mentions.

    We keep hashtags and punctuation because they may carry emotional signal.
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = URL_RE.sub("", text)

    # Replace user mentions with a generic token
    text = MENTION_RE.sub("@USER", text)

    # Strip common RT prefix
    text = re.sub(r"^RT\s+@USER:?\s*", "", text)

    # Collapse whitespace
    text = WHITESPACE_RE.sub(" ", text).strip()

    return text

def load_trump(path: Path) -> pd.DataFrame:
    """Load Trump tweets from tweets.csv.

    Expected columns (from your file):
    - id, text, isRetweet, isDeleted, device, favorites, retweets, date, isFlagged
    """
    df = pd.read_csv(path)

    # Only keep non-deleted original tweets
    if "isDeleted" in df.columns:
        df = df[df["isDeleted"] == "f"]
    if "isRetweet" in df.columns:
        df = df[df["isRetweet"] == "f"]

    df["timestamp"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["text"] = df["text"].astype(str).map(clean_text)

    df = df.dropna(subset=["timestamp", "text"])
    df["speaker"] = TRUMP
    df["source"] = "trump_csv"
    df["row_id"] = df.index.astype(str)

    return df[["speaker", "timestamp", "text", "source", "row_id"]]

def load_obama(path: Path) -> pd.DataFrame:
    """Load Obama tweets from obama.csv.

    Expected columns (from your file):
    - Text, Timestamp, Embedded_text, ...

    We use `Embedded_text` as the tweet content when available,
    otherwise we fall back to `Text`.
    """
    df = pd.read_csv(path)

    if "Embedded_text" in df.columns:
        text_series = df["Embedded_text"].fillna("")
    else:
        text_series = df["Text"].fillna("")

    header_series = df["Text"].fillna("") if "Text" in df.columns else ""

    # Some rows put meta-info like "Barack Obama @BarackObama · May 7, 2007" in Text.
    # We prefer Embedded_text when present.
    df["text"] = text_series.where(text_series.str.len() > 0, header_series)
    df["text"] = df["text"].astype(str).map(clean_text)

    df["timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "text"])

    df["speaker"] = OBAMA
    df["source"] = "obama_csv"
    df["row_id"] = df.index.astype(str)

    return df[["speaker", "timestamp", "text", "source", "row_id"]]

def load_biden(path: Path) -> pd.DataFrame:
    """Load Biden tweets from JoeBiden.csv.

    Expected columns (from your file):
    - content, date, retweetedTweet, quotedTweet, ...

    We treat `content` as the tweet text and drop retweets.
    """
    df = pd.read_csv(path)

    # Drop retweets if that column exists
    if "retweetedTweet" in df.columns:
        df = df[df["retweetedTweet"].isna()]

    df["timestamp"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["text"] = df["content"].astype(str).map(clean_text)

    df = df.dropna(subset=["timestamp", "text"])

    df["speaker"] = BIDEN
    df["source"] = "biden_csv"
    df["row_id"] = df.index.astype(str)

    return df[["speaker", "timestamp", "text", "source", "row_id"]]

def balance_by_actor_phase(df: pd.DataFrame, max_per_actor_phase: int = 1500) -> pd.DataFrame:
    """Downsample so that no (speaker, phase) has more than max_per_actor_phase tweets."""
    if "phase" not in df.columns:
        raise ValueError("DataFrame must have a 'phase' column before balancing.")

    pieces = []
    for (speaker, phase), group in df.groupby(["speaker", "phase"], dropna=False):
        if phase is None or phase != phase:  # NaN check
            continue
        n = len(group)
        target = min(n, max_per_actor_phase)
        if target <= 0:
            continue
        sampled = group.sample(n=target, random_state=42) if target < n else group
        pieces.append(sampled)

    if not pieces:
        return df.iloc[0:0].copy()

    balanced = pd.concat(pieces, ignore_index=True)
    return balanced
