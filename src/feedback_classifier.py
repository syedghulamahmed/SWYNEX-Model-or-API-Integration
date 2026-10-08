from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_feedback.csv"


def build_model() -> Pipeline:
    """Create the text classification pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def load_data(path: Path = DATA_PATH) -> tuple[pd.Series, pd.Series]:
    """Load and validate the labeled feedback dataset."""
    data = pd.read_csv(path)
    required = {"text", "label"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if data[["text", "label"]].isna().any().any():
        raise ValueError("Dataset contains missing text or label values.")
    return data["text"], data["label"]


def train_model() -> tuple[Pipeline, dict[str, float]]:
    """Train the classifier and evaluate it on a held-out test split."""
    texts, labels = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
    }
    print("Evaluation")
    print("----------")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro-F1: {metrics['macro_f1']:.3f}")
    print()
    print(classification_report(y_test, predictions, zero_division=0))
    return model, metrics


def predict(model: Pipeline, text: str) -> tuple[str, float]:
    """Predict a category and return its highest class probability."""
    probabilities = model.predict_proba([text])[0]
    index = int(probabilities.argmax())
    return str(model.classes_[index]), float(probabilities[index])


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify student/internship feedback.")
    parser.add_argument("--text", help="Feedback text to classify.")
    args = parser.parse_args()

    model, _ = train_model()

    examples = (
        [args.text]
        if args.text
        else [
            "The dashboard is easy to use and the instructions are very clear.",
            "Could you explain how I submit my weekly task?",
            "Please add a calendar showing all upcoming deadlines.",
            "My assignment was marked late even though I submitted it on time.",
            "The portal is confusing and keeps loading when I open my tasks.",
        ]
    )

    print("Predictions")
    print("-----------")
    for text in examples:
        label, confidence = predict(model, text)
        print(f"Input: {text}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.3f}")
        print()


if __name__ == "__main__":
    main()
