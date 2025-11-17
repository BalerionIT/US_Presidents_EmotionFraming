from __future__ import annotations

from pathlib import Path
from typing import Tuple

import shap

from .emotion_model import GoEmotionsClassifier


def explain_single_text_to_html(
    text: str,
    pred_label: str | None = None,
    out_path: Path | None = None,
) -> Tuple[Path, str]:
    """Use SHAP to explain a single tweet and save as an HTML file.

    Args:
        text: The tweet text to explain.
        pred_label: The emotion label to focus on. If None, we will use the
                    classifier's top-1 prediction.
        out_path: Optional explicit HTML path. If None, uses
                  figures/shap_expl.html in the current working directory.

    Returns:
        (path, label) where label is the emotion that was explained.
    """
    clf = GoEmotionsClassifier()

    # Get model prediction if needed
    if pred_label is None:
        preds, _ = clf.predict_texts([text], batch_size=1)
        pred_label = preds[0]

    # Build SHAP explainer on top of the Hugging Face pipeline
    explainer = shap.Explainer(clf.pipe)

    shap_values = explainer([text])

    # shap_values has shape [samples, tokens, classes]. We select the class
    # corresponding to pred_label, using output_names from SHAP.
    output_names = list(shap_values.output_names)
    try:
        target_idx = output_names.index(pred_label)
    except ValueError:
        # Fallback: use the first class if mapping fails
        target_idx = 0

    # Select the explanation for the chosen class and the single sample
    class_explanation = shap_values[:, :, target_idx]

    html = shap.plots.text(class_explanation[0], display=False)

    if out_path is None:
        out_path = Path("figures") / "shap_expl.html"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    return out_path, pred_label
