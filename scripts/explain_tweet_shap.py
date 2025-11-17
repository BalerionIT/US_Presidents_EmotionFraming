import argparse
from pathlib import Path
import sys

import pandas as pd

# ---------------------------------------------------------------------
# Make sure we can import src.* when this script is run from scripts/
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainability import explain_single_text_to_html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index in data/processed/predictions_balanced.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="figures/shap_expl.html",
        help="Output HTML path",
    )
    args = parser.parse_args()

    # Load predictions
    df = pd.read_csv("data/processed/predictions_balanced.csv")
    if not (0 <= args.index < len(df)):
        raise IndexError(f"index {args.index} out of range (0..{len(df) - 1})")

    row = df.iloc[args.index]
    text = str(row["text"])
    label = str(row["predicted_emotion"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Explaining row {args.index}: "
        f"speaker={row['speaker']}, phase={row['phase']}, "
        f"key_emotion={row['key_emotion']}, label={label}"
    )

    explain_single_text_to_html(text, label, out_path)
    print(f"Saved SHAP HTML to {out_path}")


if __name__ == "__main__":
    main()