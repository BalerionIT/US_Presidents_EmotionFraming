import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"


def attention_heatmap(text: str, out_path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        output_attentions=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    attn = outputs.attentions[-1][0, 0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    ax.set_title("Attention heatmap (last layer, head 0)")
    fig.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        help="Text to visualize attention for. If omitted, use a tweet from the CSV.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index from predictions_balanced.csv if --text not provided.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="figures/attention_heatmap_example.png",
    )
    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        import pandas as pd

        df = pd.read_csv("data/processed/predictions_balanced.csv")
        text = str(df.loc[args.index, "text"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    attention_heatmap(text, out_path)
    print(f"Saved attention heatmap to {out_path}")


if __name__ == "__main__":
    main()
