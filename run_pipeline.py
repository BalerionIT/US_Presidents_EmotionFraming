from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import TRUMP, OBAMA, BIDEN
from src.data_utils import load_trump, load_obama, load_biden, balance_by_actor_phase
from src.phases import assign_phase
from src.emotion_model import GoEmotionsClassifier
from src.analysis import add_valence_and_key, make_all_aggregates_and_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emotional framing pipeline for Obama / Trump / Biden tweets (GoEmotions-based)."
    )
    parser.add_argument(
        "--trump_csv",
        type=Path,
        default=Path("data/raw/tweets.csv"),
        help="Path to Trump tweets CSV (default: data/raw/tweets.csv).",
    )
    parser.add_argument(
        "--obama_csv",
        type=Path,
        default=Path("data/raw/obama.csv"),
        help="Path to Obama tweets CSV (default: data/raw/obama.csv).",
    )
    parser.add_argument(
        "--biden_csv",
        type=Path,
        default=Path("data/raw/JoeBiden.csv"),
        help="Path to Biden tweets CSV (default: data/raw/JoeBiden.csv).",
    )
    parser.add_argument(
        "--max_per_actor_phase",
        type=int,
        default=1500,
        help="Maximum tweets per (speaker, phase) after balancing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for transformer inference.",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed CSV outputs.",
    )
    parser.add_argument(
        "--figures_dir",
        type=Path,
        default=Path("figures"),
        help="Directory for PNG figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load all three datasets
    print("Loading Trump tweets...")
    trump_df = load_trump(args.trump_csv)
    print(f"  Loaded {len(trump_df)} Trump tweets.")

    print("Loading Obama tweets...")
    obama_df = load_obama(args.obama_csv)
    print(f"  Loaded {len(obama_df)} Obama tweets.")

    print("Loading Biden tweets...")
    biden_df = load_biden(args.biden_csv)
    print(f"  Loaded {len(biden_df)} Biden tweets.")

    df = pd.concat([trump_df, obama_df, biden_df], ignore_index=True)

    # Filter out extremely short tweets (less than 5 tokens)
    df = df[df["text"].str.split().str.len() >= 5].copy()
    print(f"After min-length filter: {len(df)} tweets.")

    # 2. Assign phases
    print("Assigning phases (campaign / presidency)...")
    df["phase"] = df.apply(lambda row: assign_phase(row["speaker"], row["timestamp"]), axis=1)
    from src.config import PHASE_CAMPAIGN, PHASE_PRESIDENCY
    before = len(df)
    df = df[df["phase"].isin([PHASE_CAMPAIGN, PHASE_PRESIDENCY])].copy()
    print(f"Keeping only campaign and presidency phases: {len(df)} of {before} tweets.")

    # 3. Balance across (speaker, phase)
    print("Balancing by (speaker, phase)...")
    balanced = balance_by_actor_phase(df, max_per_actor_phase=args.max_per_actor_phase)
    print("Counts after balancing:")
    print(balanced.groupby(["speaker", "phase"]).size())

    # 4. Emotion classification with GoEmotions model
    print("Loading GoEmotions classifier...")
    classifier = GoEmotionsClassifier()

    print("Running emotion inference...")
    preds, top3 = classifier.predict_texts(
        balanced["text"].tolist(),
        batch_size=args.batch_size,
        max_length=128,
    )
    balanced["predicted_emotion"] = preds

    # Optionally store top-3 scores as a string (to keep CSV readable)
    import json

    balanced["top3_emotions"] = [
        json.dumps(
            [{"label": d["label"], "score": round(float(d["score"]), 4)} for d in lst]
        )
        for lst in top3
    ]

    # 5. Add valence and key emotions
    print("Adding valence and key_emotion columns...")
    balanced = add_valence_and_key(balanced)
    # Add rhetorical target and topic
    from src.rhetoric import tag_target_type, tag_topic
    print("Tagging rhetorical target_type and topic...")
    balanced["target_type"] = balanced["text"].map(tag_target_type)
    balanced["topic"] = balanced["text"].map(tag_topic)

    # 6. Save full predictions
    processed_dir: Path = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    pred_path = processed_dir / "predictions_balanced.csv"
    balanced.to_csv(pred_path, index=False)
    print(f"Saved predictions to: {pred_path}")

    # 7. Aggregates + plots
    figures_dir: Path = args.figures_dir
    emo, val = make_all_aggregates_and_plots(balanced, processed_dir, figures_dir)
    print(f"Saved aggregated CSVs to: {processed_dir}")
    print(f"Saved figures to: {figures_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
