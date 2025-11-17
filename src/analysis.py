from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd

# Map fine-grained GoEmotions labels to coarse valence
POSITIVE = {
    "admiration",
    "amusement",
    "approval",
    "caring",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
}
NEGATIVE = {
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
}

KEY_EMOTIONS = [
    "anger",
    "fear",
    "pride",
    "joy",
    "sadness",
    "optimism",
    "gratitude",
]


def add_valence_and_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'valence' and 'key_emotion' columns based on predicted_emotion."""

    def valence(label: str) -> str:
        if label in POSITIVE:
            return "positive"
        if label in NEGATIVE:
            return "negative"
        if label == "neutral":
            return "neutral"
        return "neutral/other"

    def key_emotion(label: str) -> str:
        return label if label in KEY_EMOTIONS else "other"

    df = df.copy()
    df["valence"] = df["predicted_emotion"].map(valence)
    df["key_emotion"] = df["predicted_emotion"].map(key_emotion)
    return df


def _save_agg_csvs(
    df: pd.DataFrame,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Emotion distribution
    emo = (
        df.groupby(["speaker", "phase", "predicted_emotion"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        df.groupby(["speaker", "phase"])
        .size()
        .reset_index(name="total")
    )
    emo = emo.merge(totals, on=["speaker", "phase"])
    emo["share"] = emo["count"] / emo["total"].clip(lower=1)
    emo.to_csv(out_dir / "agg_emotions.csv", index=False)

    # Valence distribution
    val = (
        df.groupby(["speaker", "phase", "valence"])
        .size()
        .reset_index(name="count")
    )
    vtot = (
        df.groupby(["speaker", "phase"])
        .size()
        .reset_index(name="total")
    )
    val = val.merge(vtot, on=["speaker", "phase"])
    val["share"] = val["count"] / val["total"].clip(lower=1)
    val.to_csv(out_dir / "agg_valence.csv", index=False)

    return emo, val


def _plot_valence_by_actor_phase(val: pd.DataFrame, out_path: Path) -> None:
    if val.empty:
        return

    pivot = val.pivot_table(
        index=["speaker", "phase"],
        columns="valence",
        values="share",
        fill_value=0.0,
    )

    # Stable column order
    col_order = ["positive", "negative", "neutral", "neutral/other"]
    cols = [c for c in col_order if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(8, 5))

    indices = range(len(pivot))
    bottoms = [0.0] * len(pivot)

    for col in cols:
        values = pivot[col].values
        ax.bar(indices, values, bottom=bottoms, label=col)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_xticks(list(indices))
    ax.set_xticklabels(
        [f"{s}\n{p}" for s, p in pivot.index],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Share of tweets")
    ax.set_title("Valence distribution by actor and phase")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_key_emotions_by_actor_phase(df: pd.DataFrame, out_path: Path) -> None:
    subset = df[df["key_emotion"].isin(KEY_EMOTIONS)]
    if subset.empty:
        return

    agg = (
        subset.groupby(["speaker", "phase", "key_emotion"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        subset.groupby(["speaker", "phase"])
        .size()
        .reset_index(name="total")
    )
    agg = agg.merge(totals, on=["speaker", "phase"])
    agg["share"] = agg["count"] / agg["total"].clip(lower=1)

    # Create a grouped bar chart: x-axis = (speaker, phase), grouped by key_emotion
    pivot = agg.pivot_table(
        index=["speaker", "phase"],
        columns="key_emotion",
        values="share",
        fill_value=0.0,
    )
    cols = [c for c in KEY_EMOTIONS if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(9, 5))

    indices = range(len(pivot))
    width = 0.8 / max(1, len(cols))

    for i, col in enumerate(cols):
        offsets = [x + (i - len(cols) / 2) * width + width / 2 for x in indices]
        ax.bar(offsets, pivot[col].values, width=width, label=col)

    ax.set_xticks(list(indices))
    ax.set_xticklabels(
        [f"{s}\n{p}" for s, p in pivot.index],
        rotation=45,
        ha="right",
    )
    ax.set_ylabel("Share within key emotions")
    ax.set_title("Key emotions by actor and phase")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_anger_fear_over_time(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return

    df = df.copy()
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

    subset = df[df["predicted_emotion"].isin(["anger", "fear"])]
    if subset.empty:
        return

    agg = (
        subset.groupby(["speaker", "month", "predicted_emotion"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        df.groupby(["speaker", "month"])
        .size()
        .reset_index(name="total")
    )
    agg = agg.merge(totals, on=["speaker", "month"])
    agg["share"] = agg["count"] / agg["total"].clip(lower=1)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot anger and fear separately for each speaker
    for speaker in sorted(agg["speaker"].unique()):
        for emo in ["anger", "fear"]:
            tmp = agg[(agg["speaker"] == speaker) & (agg["predicted_emotion"] == emo)]
            if tmp.empty:
                continue
            ax.plot(tmp["month"], tmp["share"], marker="o", linestyle="-", label=f"{speaker} - {emo}")

    ax.set_ylabel("Share of tweets")
    ax.set_title("Anger and fear over time (monthly)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_positive_emotions_over_time(df: pd.DataFrame, out_path: Path) -> None:
    """Plot positive key emotions (pride, joy, optimism, gratitude) over time by actor."""
    if df.empty:
        return

    df = df.copy()
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()

    pos_labels = ["pride", "joy", "optimism", "gratitude"]
    subset = df[df["predicted_emotion"].isin(pos_labels)]
    if subset.empty:
        return

    agg = (
        subset.groupby(["speaker", "month", "predicted_emotion"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        df.groupby(["speaker", "month"])
        .size()
        .reset_index(name="total")
    )
    agg = agg.merge(totals, on=["speaker", "month"])
    agg["share"] = agg["count"] / agg["total"].clip(lower=1)

    fig, ax = plt.subplots(figsize=(9, 5))

    for speaker in sorted(agg["speaker"].unique()):
        for emo in pos_labels:
            tmp = agg[(agg["speaker"] == speaker) & (agg["predicted_emotion"] == emo)]
            if tmp.empty:
                continue
            ax.plot(tmp["month"], tmp["share"], marker="o", linestyle="-", label=f"{speaker} - {emo}")

    ax.set_ylabel("Share of tweets")
    ax.set_title("Positive emotions over time (monthly)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _save_rhetoric_csvs(df: pd.DataFrame, out_dir: Path) -> None:
    """Save aggregates by target_type and topic for rhetorical analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if "target_type" in df.columns:
        tgt = (
            df.groupby(["speaker", "phase", "target_type", "key_emotion"])
            .size()
            .reset_index(name="count")
        )
        totals = (
            df.groupby(["speaker", "phase", "target_type"])
            .size()
            .reset_index(name="total")
        )
        tgt = tgt.merge(totals, on=["speaker", "phase", "target_type"])
        tgt["share"] = tgt["count"] / tgt["total"].clip(lower=1)
        tgt.to_csv(out_dir / "agg_rhetoric_target.csv", index=False)

    if "topic" in df.columns:
        top = (
            df.groupby(["speaker", "phase", "topic", "key_emotion"])
            .size()
            .reset_index(name="count")
        )
        totals = (
            df.groupby(["speaker", "phase", "topic"])
            .size()
            .reset_index(name="total")
        )
        top = top.merge(totals, on=["speaker", "phase", "topic"])
        top["share"] = top["count"] / top["total"].clip(lower=1)
        top.to_csv(out_dir / "agg_topics.csv", index=False)


def _plot_key_emotions_by_target(df: pd.DataFrame, out_path: Path) -> None:
    """Plot key emotions by rhetorical target_type (self/opponent/issue)."""
    if "target_type" not in df.columns or df.empty:
        return

    subset = df[df["key_emotion"].isin(KEY_EMOTIONS)].copy()
    if subset.empty:
        return

    agg = (
        subset.groupby(["target_type", "key_emotion"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        subset.groupby(["target_type"])
        .size()
        .reset_index(name="total")
    )
    agg = agg.merge(totals, on="target_type")
    agg["share"] = agg["count"] / agg["total"].clip(lower=1)

    pivot = agg.pivot_table(
        index="target_type",
        columns="key_emotion",
        values="share",
        fill_value=0.0,
    )
    cols = [c for c in KEY_EMOTIONS if c in pivot.columns]
    if pivot.empty or not cols:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    indices = range(len(pivot))
    width = 0.8 / max(1, len(cols))

    for i, col in enumerate(cols):
        offsets = [x + (i - len(cols) / 2) * width + width / 2 for x in indices]
        ax.bar(offsets, pivot[col].values, width=width, label=col)

    ax.set_xticks(list(indices))
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.set_ylabel("Share within target type")
    ax.set_title("Key emotions by rhetorical target (self vs opponent vs issue)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_key_emotions_by_topic(df: pd.DataFrame, out_path: Path) -> None:
    """Plot key emotions by coarse topic (economy, healthcare, etc.)."""
    if "topic" not in df.columns or df.empty:
        return

    subset = df[df["key_emotion"].isin(KEY_EMOTIONS)].copy()
    if subset.empty:
        return

    agg = (
        subset.groupby(["topic", "key_emotion"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        subset.groupby(["topic"])
        .size()
        .reset_index(name="total")
    )
    agg = agg.merge(totals, on="topic")
    agg["share"] = agg["count"] / agg["total"].clip(lower=1)

    pivot = agg.pivot_table(
        index="topic",
        columns="key_emotion",
        values="share",
        fill_value=0.0,
    )
    cols = [c for c in KEY_EMOTIONS if c in pivot.columns]
    if pivot.empty or not cols:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    indices = range(len(pivot))
    width = 0.8 / max(1, len(cols))

    for i, col in enumerate(cols):
        offsets = [x + (i - len(cols) / 2) * width + width / 2 for x in indices]
        ax.bar(offsets, pivot[col].values, width=width, label=col)

    ax.set_xticks(list(indices))
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("Share within topic")
    ax.set_title("Key emotions by topic")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_key_emotions_by_target_by_actor(df: pd.DataFrame, out_path: Path) -> None:
    """Plot key emotions by rhetorical target, with a separate panel for each actor."""
    if "target_type" not in df.columns or "speaker" not in df.columns or df.empty:
        return

    subset = df[df["key_emotion"].isin(KEY_EMOTIONS)].copy()
    if subset.empty:
        return

    speakers = sorted(subset["speaker"].unique())
    if not speakers:
        return

    fig, axes = plt.subplots(
        1, len(speakers),
        figsize=(4 * len(speakers), 4),
        sharey=True,
    )
    if len(speakers) == 1:
        axes = [axes]

    for ax, speaker in zip(axes, speakers):
        sub = subset[subset["speaker"] == speaker]
        if sub.empty:
            ax.set_title(speaker)
            ax.axis("off")
            continue

        agg = (
            sub.groupby(["target_type", "key_emotion"])
            .size()
            .reset_index(name="count")
        )
        totals = (
            sub.groupby(["target_type"])
            .size()
            .reset_index(name="total")
        )
        agg = agg.merge(totals, on="target_type")
        agg["share"] = agg["count"] / agg["total"].clip(lower=1)

        pivot = agg.pivot_table(
            index="target_type",
            columns="key_emotion",
            values="share",
            fill_value=0.0,
        )
        cols = [c for c in KEY_EMOTIONS if c in pivot.columns]
        if pivot.empty or not cols:
            ax.set_title(speaker)
            ax.axis("off")
            continue

        indices = range(len(pivot))
        width = 0.8 / max(1, len(cols))

        for i, col in enumerate(cols):
            offsets = [x + (i - len(cols) / 2) * width + width / 2 for x in indices]
            ax.bar(offsets, pivot[col].values, width=width, label=col if ax is axes[0] else None)

        ax.set_xticks(list(indices))
        ax.set_xticklabels(list(pivot.index), rotation=0)
        ax.set_title(speaker)
        ax.set_xlabel("Target type")

    axes[0].set_ylabel("Share within target type")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(labels), 4),
        )

    fig.suptitle("Key emotions by rhetorical target, by actor", y=0.98)
    fig.tight_layout(rect=[0.0, 0.15, 1.0, 0.90])

    fig.savefig(out_path)
    plt.close(fig)



def _plot_key_emotions_by_topic_by_actor(df: pd.DataFrame, out_path: Path) -> None:
    """Plot key emotions by topic, with a separate panel for each actor."""
    if "topic" not in df.columns or "speaker" not in df.columns or df.empty:
        return

    subset = df[df["key_emotion"].isin(KEY_EMOTIONS)].copy()
    if subset.empty:
        return

    speakers = sorted(subset["speaker"].unique())
    if not speakers:
        return

    fig, axes = plt.subplots(
        1, len(speakers),
        figsize=(4 * len(speakers), 4),
        sharey=True,
    )
    if len(speakers) == 1:
        axes = [axes]

    for ax, speaker in zip(axes, speakers):
        sub = subset[subset["speaker"] == speaker]
        if sub.empty:
            ax.set_title(speaker)
            ax.axis("off")
            continue

        agg = (
            sub.groupby(["topic", "key_emotion"])
            .size()
            .reset_index(name="count")
        )
        totals = (
            sub.groupby(["topic"])
            .size()
            .reset_index(name="total")
        )
        agg = agg.merge(totals, on="topic")
        agg["share"] = agg["count"] / agg["total"].clip(lower=1)

        pivot = agg.pivot_table(
            index="topic",
            columns="key_emotion",
            values="share",
            fill_value=0.0,
        )
        cols = [c for c in KEY_EMOTIONS if c in pivot.columns]
        if pivot.empty or not cols:
            ax.set_title(speaker)
            ax.axis("off")
            continue

        indices = range(len(pivot))
        width = 0.8 / max(1, len(cols))

        for i, col in enumerate(cols):
            offsets = [x + (i - len(cols) / 2) * width + width / 2 for x in indices]
            ax.bar(offsets, pivot[col].values, width=width, label=col if ax is axes[0] else None)

        ax.set_xticks(list(indices))
        ax.set_xticklabels(list(pivot.index), rotation=45, ha="right")
        ax.set_title(speaker)
        ax.set_xlabel("Topic")

    axes[0].set_ylabel("Share within topic")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(len(labels), 4),
        )
    # Put the suptitle slightly above the subplots
    fig.suptitle("Key emotions by topic, by actor", y=0.98)

    # Leave room at top for suptitle and at bottom for legend
    fig.tight_layout(rect=[0.0, 0.15, 1.0, 0.90])

    fig.savefig(out_path)
    plt.close(fig)



def _heatmap_emotions_by_topic_by_actor(df: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of key emotions by topic, with one panel per actor."""
    if "topic" not in df.columns or df.empty:
        return

    subset = df[df["key_emotion"].isin(KEY_EMOTIONS)].copy()
    if subset.empty:
        return

    speakers = sorted(subset["speaker"].unique())
    topics = [
        "economy",
        "environment",
        "foreign_policy",
        "healthcare",
        "immigration",
        "other",
        "rights_justice",
        "security",
    ]

    import numpy as np
    import matplotlib.pyplot as plt

    # extra width so we can leave a big margin on the right for the colorbar
    fig, axes = plt.subplots(
        1,
        len(speakers),
        figsize=(4 * len(speakers) + 2, 4),
        sharey=True,
    )
    if len(speakers) == 1:
        axes = [axes]

    vmin, vmax = 0.0, 0.0
    pivots = {}

    for speaker in speakers:
        sub = subset[subset["speaker"] == speaker]
        if sub.empty:
            continue

        agg = (
            sub.groupby(["topic", "key_emotion"])
            .size()
            .reset_index(name="count")
        )
        totals = (
            sub.groupby(["topic"])
            .size()
            .reset_index(name="total")
        )
        agg = agg.merge(totals, on="topic")
        agg["share"] = agg["count"] / agg["total"].clip(lower=1)

        pivot = agg.pivot_table(
            index="topic",
            columns="key_emotion",
            values="share",
            fill_value=0.0,
        )
        pivot = pivot.reindex(
            index=topics,
            columns=[k for k in KEY_EMOTIONS if k in pivot.columns],
        )
        pivots[speaker] = pivot
        if not pivot.empty:
            vmax = max(vmax, float(pivot.values.max()))

    if vmax == 0.0:
        return

    last_im = None
    for ax, speaker in zip(axes, speakers):
        pivot = pivots.get(speaker)
        ax.set_title(speaker)
        if pivot is None or pivot.empty:
            ax.axis("off")
            continue

        last_im = ax.imshow(
            pivot.values,
            aspect="auto",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Key emotion")
        if ax is axes[0]:
            ax.set_ylabel("Topic")

    if last_im is None:
        return

    # big margin on the right, colorbar attached to all axes
    fig.suptitle("Key emotions by topic (heatmap, by actor)")
    fig.tight_layout(rect=[0.03, 0.05, 0.86, 0.90])

    cbar = fig.colorbar(
        last_im,
        ax=axes,
        location="right",
        fraction=0.04,
        pad=0.02,
    )
    cbar.set_label("Share within topic")

    fig.savefig(out_path)
    plt.close(fig)


def _heatmap_emotions_by_target_by_actor(df: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of key emotions by rhetorical target_type, with one panel per actor."""
    if "target_type" not in df.columns or df.empty:
        return

    subset = df[df["key_emotion"].isin(KEY_EMOTIONS)].copy()
    if subset.empty:
        return

    speakers = sorted(subset["speaker"].unique())
    targets = ["issue", "opponent", "self", "other"]

    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(speakers),
        figsize=(4 * len(speakers) + 2, 4),
        sharey=True,
    )
    if len(speakers) == 1:
        axes = [axes]

    vmin, vmax = 0.0, 0.0
    pivots = {}

    for speaker in speakers:
        sub = subset[subset["speaker"] == speaker]
        if sub.empty:
            continue

        agg = (
            sub.groupby(["target_type", "key_emotion"])
            .size()
            .reset_index(name="count")
        )
        totals = (
            sub.groupby(["target_type"])
            .size()
            .reset_index(name="total")
        )
        agg = agg.merge(totals, on="target_type")
        agg["share"] = agg["count"] / agg["total"].clip(lower=1)

        pivot = agg.pivot_table(
            index="target_type",
            columns="key_emotion",
            values="share",
            fill_value=0.0,
        )
        pivot = pivot.reindex(
            index=targets,
            columns=[k for k in KEY_EMOTIONS if k in pivot.columns],
        )
        pivots[speaker] = pivot
        if not pivot.empty:
            vmax = max(vmax, float(pivot.values.max()))

    if vmax == 0.0:
        return

    last_im = None
    for ax, speaker in zip(axes, speakers):
        pivot = pivots.get(speaker)
        ax.set_title(speaker)
        if pivot is None or pivot.empty:
            ax.axis("off")
            continue

        last_im = ax.imshow(
            pivot.values,
            aspect="auto",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Key emotion")
        if ax is axes[0]:
            ax.set_ylabel("Target type")

    if last_im is None:
        return

    fig.suptitle("Key emotions by rhetorical target (heatmap, by actor)")
    fig.tight_layout(rect=[0.03, 0.05, 0.86, 0.90])

    cbar = fig.colorbar(
        last_im,
        ax=axes,
        location="right",
        fraction=0.04,
        pad=0.02,
    )
    cbar.set_label("Share within target_type")

    fig.savefig(out_path)
    plt.close(fig)


def make_all_aggregates_and_plots(
    df: pd.DataFrame,
    processed_dir: Path,
    figures_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create aggregate CSVs and main PNG plots."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    emo, val = _save_agg_csvs(df, processed_dir)
    _save_rhetoric_csvs(df, processed_dir)

    _plot_valence_by_actor_phase(val, figures_dir / "valence_by_actor_phase.png")
    _plot_key_emotions_by_actor_phase(df, figures_dir / "key_emotions_by_actor_phase.png")
    _plot_anger_fear_over_time(df, figures_dir / "anger_fear_over_time.png")
    _plot_positive_emotions_over_time(df, figures_dir / "positive_emotions_over_time.png")
    _plot_key_emotions_by_target(df, figures_dir / "key_emotions_by_target.png")
    _plot_key_emotions_by_topic(df, figures_dir / "key_emotions_by_topic.png")
    _plot_key_emotions_by_target_by_actor(df, figures_dir / "key_emotions_by_target_by_actor.png")
    _plot_key_emotions_by_topic_by_actor(df, figures_dir / "key_emotions_by_topic_by_actor.png")
    _heatmap_emotions_by_target_by_actor(df, figures_dir / "heatmap_emotions_by_target_by_actor.png")
    _heatmap_emotions_by_topic_by_actor(df, figures_dir / "heatmap_emotions_by_topic_by_actor.png")

    return emo, val
