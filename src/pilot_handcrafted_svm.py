import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)

from load_esc50 import load_metadata
from audio_utils import load_audio, crop_audio
from handcrafted_features import extract_handcrafted_features


DURATIONS = [1, 2, 5]
SEEDS = [42, 43, 44, 45, 46]


def build_features(df, duration, seed):
    X = []
    y_labels = []

    for idx, row in df.iterrows():
        audio, sr = load_audio(row["filepath"])

        if duration < 5:
            crop_seed = seed + int(idx)

            audio = crop_audio(
                audio,
                sr,
                duration=duration,
                seed=crop_seed,
            )

        features = extract_handcrafted_features(
            audio,
            sr,
        )

        X.append(features)
        y_labels.append(row["category"])

    return np.array(X), np.array(y_labels)


def run_one_experiment(
    train_df,
    test_df,
    duration,
    seed,
):
    X_train, y_train = build_features(
        train_df,
        duration,
        seed,
    )

    X_test, y_test = build_features(
        test_df,
        duration,
        seed,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="linear",
            C=1.0,
        )),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    classes = sorted(
        np.unique(y_test)
    )

    per_class_recall = recall_score(
        y_test,
        predictions,
        labels=classes,
        average=None,
    )

    return (
        accuracy,
        macro_f1,
        classes,
        per_class_recall,
    )


if __name__ == "__main__":
    train_df, test_df = load_metadata()

    results = []
    recall_results = []

    for duration in DURATIONS:

        seeds = (
            SEEDS
            if duration < 5
            else [42]
        )

        for seed in seeds:

            print(
                f"Running duration={duration}s, "
                f"seed={seed}"
            )

            (
                accuracy,
                macro_f1,
                classes,
                recalls,
            ) = run_one_experiment(
                train_df,
                test_df,
                duration,
                seed,
            )

            results.append({
                "duration": duration,
                "seed": seed,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            })

            for class_name, recall in zip(
                classes,
                recalls,
            ):
                recall_results.append({
                    "duration": duration,
                    "seed": seed,
                    "category": class_name,
                    "recall": recall,
                })

    results_df = pd.DataFrame(
        results
    )

    print("\nIndividual runs:")
    print(results_df)

    summary = (
        results_df
        .groupby("duration")[
            ["accuracy", "macro_f1"]
        ]
        .agg(["mean", "std"])
    )

    print("\nSummary:")
    print(summary)

    results_df.to_csv(
        "results/pilot_handcrafted_results.csv",
        index=False,
    )

    recall_df = pd.DataFrame(
        recall_results
    )

    recall_df.to_csv(
        "results/"
        "pilot_handcrafted_per_class_recall.csv",
        index=False,
    )

    print(
        "\nSaved to "
        "results/pilot_handcrafted_results.csv"
    )

    print(
        "Saved per-class recall to "
        "results/"
        "pilot_handcrafted_per_class_recall.csv"
    )