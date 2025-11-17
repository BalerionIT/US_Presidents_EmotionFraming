# Emotional Framing in US Presidential Tweets (Twitter-Only Pipeline)

This project analyzes how **emotional language** is used strategically in tweets from three recent US presidents:
**Barack Obama, Donald Trump, and Joe Biden**.

It uses a **pre-trained transformer model trained on the GoEmotions dataset** to classify emotions in tweets,
then compares emotional patterns across **actors** and **phases** (campaign vs. presidency) and over time.
There is also an optional **SHAP-based explainability** script to highlight emotionally important words.

The project is designed to run end-to-end from **Visual Studio Code on macOS** (Python 3.10+).

---

## 1. Folder structure

```text
EmotionFraming_Twitter/
├─ README.md
├─ requirements.txt
├─ run_pipeline.py
├─ src/
│  ├─ __init__.py
│  ├─ config.py          # Date ranges for campaigns & presidencies
│  ├─ phases.py          # Assign phase (campaign / presidency / other)
│  ├─ data_utils.py      # Load & clean the three CSVs, balancing
│  ├─ emotion_model.py   # GoEmotions-based transformer classifier
│  ├─ analysis.py        # Aggregations and static plots
│  └─ explainability.py  # Helper for SHAP explainability (used by script)
├─ app/
│  └─ streamlit_app.py   # Simple dashboard to explore results
├─ scripts/
│  └─ explain_tweet_shap.py  # Optional: SHAP explanation for one tweet
├─ data/
│  ├─ raw/               # Put your input CSVs here
│  └─ processed/         # Predictions and aggregated CSVs
└─ figures/              # PNG plots + SHAP HTML files
```

---

## 2. Required input files (your CSVs)

Copy your three CSVs into `data/raw/` using the following names:

- Trump tweets: `data/raw/tweets.csv`  (your original `tweets.csv`)
- Obama tweets: `data/raw/obama.csv`  (your original `obama.csv`)
- Biden tweets: `data/raw/JoeBiden.csv` (your original `JoeBiden.csv`)

The code assumes exactly these filenames, so **don’t rename them** unless you also change
the `--*_csv` arguments when you run the pipeline.

---

## 3. Installation & setup in Visual Studio Code (macOS)

1. **Download and unzip the project**
   - Save the zip from ChatGPT as, e.g., `EmotionFraming_Twitter.zip`.
   - Unzip it, for instance into `~/Downloads/EmotionFraming_Twitter`.

2. **Open the folder in VS Code**
   - In Visual Studio Code: `File → Open Folder…`
   - Select the `EmotionFraming_Twitter` folder.

3. **Create and activate a virtual environment**
   Open a VS Code terminal in that folder (`Terminal → New Terminal`) and run:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # On macOS / Linux
   # On Windows PowerShell (if you ever use it):
   # .venv\Scripts\Activate.ps1
   ```

4. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   This will install `torch`, `transformers`, `pandas`, `matplotlib`, `streamlit`, `shap`, etc.
   The first time you run the model it will also download weights from Hugging Face.

5. **Place your CSVs**

   Copy your three CSVs into `data/raw/` as described above.

   ```text
   data/raw/tweets.csv       # Trump
   data/raw/obama.csv        # Obama
   data/raw/JoeBiden.csv     # Biden
   ```

---

## 4. Running the main pipeline

From the activated virtual environment in the project root:

```bash
python run_pipeline.py       --trump_csv data/raw/tweets.csv       --obama_csv data/raw/obama.csv       --biden_csv data/raw/JoeBiden.csv       --max_per_actor_phase 1500
```

What this does:

1. **Load and clean tweets**
   - Trump: filters out deleted tweets and retweets, parses `date`.
   - Obama: uses the `Embedded_text` field as tweet text, parses `Timestamp`.
   - Biden: uses `content` and drops rows where `retweetedTweet` is non-null.

2. **Standardize columns**
   - Final columns: `speaker`, `timestamp`, `text`, `source`, and an internal `row_id`.

3. **Assign phases**
   Each tweet is labeled with a `phase` based on its date:
   - `campaign` = when the politician is officially running for president (e.g. Obama 2008, 2012; Trump 2016, 2020; Biden 2020).
   - `presidency` = when they are in office as president.
   - `other` = everything else (pre-campaign, vice presidency, post-presidency).

   Campaign and presidency windows are based on publicly documented announcement and inauguration dates
   (Obama 2008 & 2012 campaigns; Trump 2016 & 2020 campaigns; Biden 2020 campaign).

4. **Balance the dataset**
   - For each `(speaker, phase)` combination, the script samples up to `max_per_actor_phase` tweets
     (default 1500) to keep runtime manageable.
   - This avoids Trump dominating just because he tweets more.

5. **Classify emotions with a GoEmotions model**
   - Uses the Hugging Face model: `joeddav/distilbert-base-uncased-go-emotions-student`
     (a distilled DistilBERT model trained on the GoEmotions dataset).
   - For each tweet, it predicts one **dominant GoEmotions label** out of the 28 classes
     (admiration, amusement, anger, fear, pride, sadness, etc., plus neutral).

   The script stores:
   - `predicted_emotion`: the label with highest probability.
   - `phase`, `speaker`, `timestamp`, `text`.

6. **Add higher-level categories**
   - `valence`: {`positive`, `negative`, `neutral/other`} based on the emotion.
   - `key_emotion`: keeps a small set of “headline” emotions (e.g. anger, fear, pride, joy, sadness, optimism, gratitude),
     grouping all other emotions as `"other"`.

7. **Save outputs**
   - `data/processed/predictions_balanced.csv` – one row per tweet with model predictions and metadata.
   - `data/processed/agg_emotions.csv` – aggregated counts and shares by actor, phase, and emotion.
   - `data/processed/agg_valence.csv` – positive/negative/neutral breakdown by actor and phase.

8. **Generate static plots (saved in `figures/`)**
   - `valence_by_actor_phase.png` – stacked bar chart of positive / negative / neutral by actor & phase.
   - `key_emotions_by_actor_phase.png` – shares of key emotions (anger, fear, pride, joy, sadness, etc.).
   - `anger_fear_over_time.png` – monthly evolution of anger vs. fear for each president.

---

## 5. Interactive dashboard (Streamlit)

After running `run_pipeline.py` at least once, launch the dashboard:

```bash
streamlit run app/streamlit_app.py
```

In the browser app you can:

- Filter by **actor** (Obama / Trump / Biden).
- Filter by **phase** (campaign / presidency / other).
- Focus on specific **emotions** (e.g. anger, fear, pride).
- See interactive bar charts and time-series of emotion usage.

This covers the “plots or dashboards comparing emotional tone across actors and phases” part of the project.

---

## 6. SHAP-based explainability (optional)

To highlight **emotionally important words** in a specific tweet, you can use the SHAP script.

1. Make sure you have already run `run_pipeline.py` so that
   `data/processed/predictions_balanced.csv` exists.

2. Run the explainability script (from the project root, venv activated):

   ```bash
   python scripts/explain_tweet_shap.py --index 0
   ```

   - `--index` selects which row (tweet) from `predictions_balanced.csv` you want to explain
     (0 = first row, 1 = second, etc.).

3. The script will:
   - Load the same GoEmotions classifier.
   - Use **SHAP** to decompose the prediction into token-level contributions.
   - Save an interactive HTML explanation into `figures/shap_expl_INDEX.html`.

   You can then open that HTML file in your browser and hover over words to see how strongly
   they contribute to the predicted emotion.

If SHAP is too slow or causes memory issues, you can skip this step; the rest of the pipeline
(classification + plots + dashboard) does not depend on it.

---

## 7. Typical workflow summary

1. Open the folder in VS Code.
2. Create and activate the virtual environment.
3. Install requirements.
4. Copy your three CSVs into `data/raw/`.
5. Run `run_pipeline.py`.
6. Inspect CSV outputs and plots in `data/processed/` and `figures/`.
7. Optionally run `streamlit run app/streamlit_app.py` for an interactive dashboard.
8. Optionally run `python scripts/explain_tweet_shap.py --index N` to get a SHAP HTML explanation
   for tweet N.

This should give you a **complete, VS-Code-ready project** focused on emotional framing in
presidential tweets, aligned with the GoEmotions taxonomy and your assignment requirements.
