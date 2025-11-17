from __future__ import annotations

from typing import List, Tuple

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)


class GoEmotionsClassifier:
    """Wrapper around a Hugging Face GoEmotions model.

    Default model: joeddav/distilbert-base-uncased-go-emotions-student
    """

    def __init__(
        self,
        model_name: str = "joeddav/distilbert-base-uncased-go-emotions-student",
    ) -> None:
        device = 0 if torch.cuda.is_available() else -1

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipe = TextClassificationPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            device=device,
        )

    def predict_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 128,
    ) -> Tuple[List[str], List[List[dict]]]:
        """Run emotion prediction on a list of texts.

        Returns:
            preds: list of top-1 emotion label per text
            top3_scores: list of length-3 lists with label+score dicts
        """
        preds: List[str] = []
        top3_scores: List[List[dict]] = []

        if not texts:
            return preds, top3_scores

        for start in tqdm(
            range(0, len(texts), batch_size),
            desc="Emotion inference",
        ):
            batch = texts[start : start + batch_size]
            outputs = self.pipe(
                batch,
                truncation=True,
                max_length=max_length,
            )

            for out in outputs:
                # out is a list of {label, score}
                sorted_scores = sorted(out, key=lambda d: d["score"], reverse=True)
                best = sorted_scores[0]
                preds.append(best["label"])
                top3_scores.append(sorted_scores[:3])

        return preds, top3_scores
