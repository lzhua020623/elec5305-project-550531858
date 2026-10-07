import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay

from load_esc50 import load_metadata
from audio_utils import load_audio, crop_audio
from handcrafted_features import extract_handcrafted_features
from yamnet_features import YAMNetFeatureExtractor


DURATION = 1
SEED = 42


def build_handcrafted_features(df):
    X = []
    y_labels = []

    for idx, row in df.iterrows():

        y, sr = load_audio(
            row["filepath"]
        )

        crop_seed = SEED + int(idx)

        y = crop_audio(
            y,
            sr,
            duration=DURATION,
            seed=crop_seed,
        )

        features = extract_handcrafted_features(
            y,
            sr,
        )

        X.append(features)
        y_labels.append(
            row["category"]
        )

    return X, y_labels


def build_yamnet_features(
    df,
    extractor,
):
    X = []
    y_labels = []

    for idx, row in df.iterrows():

        y, sr = load_audio(
            row["filepath"]
        )

        crop_seed = SEED + int(idx)

        y = crop_audio(
            y,
            sr,
            duration=DURATION,
            seed=crop_seed,
        )

        embedding = extractor.extract(
            y,
            sr,
        )

        X.append(embedding)
        y_labels.append(
            row["category"]
        )

    return X, y_labels


def train_and_predict(
    X_train,
    y_train,
    X_test,
):
    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "svm",
            SVC(
                kernel="linear",
                C=1.0,
            ),
        ),
    ])

    model.fit(
        X_train,
        y_train,
    )

    return model.predict(
        X_test
    )


if __name__ == "__main__":

    train_df, test_df = load_metadata()

    classes = sorted(
        test_df["category"].unique()
    )

    # -------------------------
    # Handcrafted
    # -------------------------

    X_train_h, y_train_h = (
        build_handcrafted_features(
            train_df
        )
    )

    X_test_h, y_test_h = (
        build_handcrafted_features(
            test_df
        )
    )

    pred_h = train_and_predict(
        X_train_h,
        y_train_h,
        X_test_h,
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test_h,
        pred_h,
        labels=classes,
        normalize="true",
        xticks_rotation=30,
    )

    plt.title(
        "1s Confusion Matrix - "
        "Handcrafted Features"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/"
        "pilot_confusion_handcrafted_1s.png",
        dpi=300,
    )

    plt.show()

    # -------------------------
    # YAMNet
    # -------------------------

    extractor = (
        YAMNetFeatureExtractor()
    )

    X_train_y, y_train_y = (
        build_yamnet_features(
            train_df,
            extractor,
        )
    )

    X_test_y, y_test_y = (
        build_yamnet_features(
            test_df,
            extractor,
        )
    )

    pred_y = train_and_predict(
        X_train_y,
        y_train_y,
        X_test_y,
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test_y,
        pred_y,
        labels=classes,
        normalize="true",
        xticks_rotation=30,
    )

    plt.title(
        "1s Confusion Matrix - "
        "Frozen YAMNet Embeddings"
    )

    plt.tight_layout()

    plt.savefig(
        "figures/"
        "pilot_confusion_yamnet_1s.png",
        dpi=300,
    )

    plt.show()

    print(
        "Saved both confusion matrices."
    )