from pathlib import Path
import pandas as pd

DATA_ROOT = Path("data/ESC-50")
META_PATH = DATA_ROOT / "meta" / "esc50.csv"
AUDIO_DIR = DATA_ROOT / "audio"

PILOT_CLASSES = [
    "rain",
    "sea_waves",
    "crackling_fire",
    "clock_tick",
    "door_wood_knock",
]

TEST_FOLD = 1


def load_metadata():
    df = pd.read_csv(META_PATH)

    df = df[df["category"].isin(PILOT_CLASSES)].copy()
    df["filepath"] = df["filename"].apply(lambda x: AUDIO_DIR / x)

    train_df = df[df["fold"] != TEST_FOLD].copy()
    test_df = df[df["fold"] == TEST_FOLD].copy()

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_metadata()

    print("Pilot classes:")
    print(PILOT_CLASSES)

    print("\nTraining samples:", len(train_df))
    print("Test samples:", len(test_df))

    print("\nTraining samples per class:")
    print(train_df["category"].value_counts().sort_index())

    print("\nTest samples per class:")
    print(test_df["category"].value_counts().sort_index())

    print("\nTraining folds:")
    print(sorted(train_df["fold"].unique()))

    print("Test fold:")
    print(sorted(test_df["fold"].unique()))

    print("\nAll audio files exist:")
    print(train_df["filepath"].apply(lambda p: p.exists()).all())
    print(test_df["filepath"].apply(lambda p: p.exists()).all())