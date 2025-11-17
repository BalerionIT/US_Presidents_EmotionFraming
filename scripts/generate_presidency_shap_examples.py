import sys
from pathlib import Path

import pandas as pd

# Make sure we can import src.*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainability import explain_single_text_to_html


SPEAKERS = {
    "Barack Obama": "obama",
    "Donald Trump": "trump",
    "Joe Biden": "biden",
}

POSITIVE_PRIORITY = ["joy", "optimism", "gratitude", "pride"]
NEGATIVE_PRIORITY = ["anger", "fear", "sadness"]


def pick_example(df: pd.DataFrame, speaker: str, valence: str) -> pd.Series | None:
    """Pick one presidency tweet for a given speaker and valence (positive/negative)."""
    sub = df[
        (df["speaker"] == speaker)
        & (df["phase"] == "presidency")
        & (df["valence"] == valence)
    ]

    if sub.empty:
        print(f"[WARN] No {valence} presidency tweets for {speaker}, skipping.")
        return None

    # Prefer a "nice" key_emotion
    if valence == "positive":
        priority = POSITIVE_PRIORITY
    else:
        priority = NEGATIVE_PRIORITY

    for emo in priority:
        tmp = sub[sub["key_emotion"] == emo]
        if not tmp.empty:
            return tmp.iloc[0]

    # Fallback: just take the first one
    return sub.iloc[0]


def main() -> None:
    df = pd.read_csv("data/processed/predictions_balanced.csv")

    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    for speaker, slug in SPEAKERS.items():
        for valence in ["positive", "negative"]:
            row = pick_example(df, speaker, valence)
            if row is None:
                continue

            idx = int(row.name)
            text = str(row["text"])
            label = str(row["predicted_emotion"])

            out_path = out_dir / f"shap_{slug}_pres_{valence}.html"

            print(
                f"\n=== {speaker} – presidency – {valence} ==="
                f"\nindex={idx}, key_emotion={row['key_emotion']}, topic={row.get('topic', 'n/a')}, "
                f"target_type={row.get('target_type', 'n/a')}"
            )
            explain_single_text_to_html(text, label, out_path)
            print(f"Saved SHAP HTML to {out_path}")


if __name__ == "__main__":
    main()