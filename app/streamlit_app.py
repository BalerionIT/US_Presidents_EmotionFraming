import pandas as pd
import streamlit as st
from pathlib import Path


DATA_PATH = Path("data/processed/predictions_balanced.csv")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def main() -> None:
    st.set_page_config(page_title="Emotional Framing in US Presidential Tweets", layout="wide")

    st.title("Emotional Framing in US Presidential Tweets")
    st.markdown(
        """
        This dashboard shows **GoEmotions-based emotion predictions** for tweets by
        Barack Obama, Donald Trump, and Joe Biden, split by **campaign vs. presidency vs. other** phases.
        """
    )

    if not DATA_PATH.exists():
        st.error(
            f"Could not find {DATA_PATH}. Run `python run_pipeline.py` first to generate predictions."
        )
        return

    df = load_data(DATA_PATH)

    # Sidebar filters
    st.sidebar.header("Filters")

    speakers = sorted(df["speaker"].unique())
    phases = sorted(df["phase"].unique())

    selected_speakers = st.sidebar.multiselect(
        "Select actors (speakers):", speakers, default=speakers
    )
    selected_phases = st.sidebar.multiselect(
        "Select phases:", phases, default=phases
    )

    emotion_options = sorted(df["predicted_emotion"].unique())
    default_emotions = [e for e in emotion_options if e in ["anger", "fear", "pride", "joy", "sadness"]] or emotion_options
    selected_emotions = st.sidebar.multiselect(
        "Focus on emotions:", emotion_options, default=default_emotions
    )

    df_filt = df[
        df["speaker"].isin(selected_speakers)
        & df["phase"].isin(selected_phases)
    ].copy()

    if df_filt.empty:
        st.warning("No tweets for this selection.")
        return

    st.subheader("Overall emotion distribution")

    emo = (
        df_filt.groupby(["speaker", "phase", "predicted_emotion"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        df_filt.groupby(["speaker", "phase"])
        .size()
        .reset_index(name="total")
    )
    emo = emo.merge(totals, on=["speaker", "phase"])
    emo["share"] = emo["count"] / emo["total"].clip(lower=1)

    # Filter to selected emotions for the chart
    emo_sel = emo[emo["predicted_emotion"].isin(selected_emotions)].copy()

    if emo_sel.empty:
        st.info("No tweets with the selected emotions for this filter combination.")
    else:
        pivot = emo_sel.pivot_table(
            index=["speaker", "phase"],
            columns="predicted_emotion",
            values="share",
            fill_value=0.0,
        )

        # Streamlit's built-in bar_chart does not like MultiIndex,
        # so we collapse (speaker, phase) into a single index column.
        pivot_chart = pivot.copy()
        pivot_chart["actor_phase"] = [f"{s}\n{p}" for s, p in pivot_chart.index]
        pivot_chart = pivot_chart.set_index("actor_phase")

        st.bar_chart(pivot_chart)

    st.subheader("Anger and fear over time")

    df_filt["month"] = df_filt["timestamp"].dt.to_period("M").dt.to_timestamp()
    subset = df_filt[df_filt["predicted_emotion"].isin(["anger", "fear"])]

    if subset.empty:
        st.info("No anger/fear tweets for this filter combination.")
    else:
        time_agg = (
            subset.groupby(["speaker", "month", "predicted_emotion"])
            .size()
            .reset_index(name="count")
        )
        totals = (
            df_filt.groupby(["speaker", "month"])
            .size()
            .reset_index(name="total")
        )
        time_agg = time_agg.merge(totals, on=["speaker", "month"])
        time_agg["share"] = time_agg["count"] / time_agg["total"].clip(lower=1)

        # For Streamlit line_chart we pivot on month
        for speaker in sorted(time_agg["speaker"].unique()):
            st.markdown(f"**{speaker}**")
            tmp = time_agg[time_agg["speaker"] == speaker]
            pivot = tmp.pivot_table(
                index="month",
                columns="predicted_emotion",
                values="share",
                fill_value=0.0,
            ).sort_index()
            st.line_chart(pivot)


if __name__ == "__main__":
    main()
