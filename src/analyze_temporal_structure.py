import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from load_esc50 import load_metadata
from audio_utils import load_audio


CONTEXT_PATH = (
    "results/pilot_context_dependence.csv"
)

OUTPUT_CSV = (
    "results/pilot_temporal_structure.csv"
)


def calculate_temporal_features(y):
    """
    Calculate temporal energy variability and spectral flux
    from one complete audio recording.
    """

    # RMS energy over short-time frames
    rms = librosa.feature.rms(y=y)[0]

    energy_variability = (
        np.std(rms) /
        (np.mean(rms) + 1e-8)
    )

    # Magnitude STFT
    magnitude = np.abs(
        librosa.stft(y)
    )

    # Difference between adjacent spectra
    spectral_diff = np.diff(
        magnitude,
        axis=1,
    )

    spectral_flux = np.sqrt(
        np.sum(
            spectral_diff ** 2,
            axis=0,
        )
    )

    mean_spectral_flux = np.mean(
        spectral_flux
    )

    return (
        energy_variability,
        mean_spectral_flux,
    )


if __name__ == "__main__":

    train_df, test_df = load_metadata()

    # Use all 40 recordings per pilot category
    df = pd.concat(
        [train_df, test_df],
        ignore_index=True,
    )

    rows = []

    for _, row in df.iterrows():

        y, sr = load_audio(
            row["filepath"]
        )

        (
            energy_variability,
            spectral_flux,
        ) = calculate_temporal_features(y)

        rows.append({
            "category": row["category"],
            "energy_variability":
                energy_variability,
            "spectral_flux":
                spectral_flux,
        })

    temporal_df = pd.DataFrame(rows)

    # Average acoustic descriptors
    # across recordings in each class
    class_summary = (
        temporal_df
        .groupby("category")[
            [
                "energy_variability",
                "spectral_flux",
            ]
        ]
        .mean()
    )

    context_df = pd.read_csv(
        CONTEXT_PATH,
        index_col=0,
    )

    combined = class_summary.join(
        context_df
    )

    combined.to_csv(
        OUTPUT_CSV
    )

    print(
        "\nTemporal structure and "
        "context dependence:"
    )
    print(combined)

    # -------------------------
    # Energy variability plot
    # -------------------------

    plt.figure(figsize=(7, 5))

    plt.scatter(
        combined["energy_variability"],
        combined["Handcrafted"],
        label="Handcrafted",
    )

    plt.scatter(
        combined["energy_variability"],
        combined["YAMNet"],
        label="YAMNet",
    )

    for category, row in combined.iterrows():

        plt.annotate(
            category,
            (
                row["energy_variability"],
                row["Handcrafted"],
            ),
            fontsize=8,
        )

    plt.xlabel(
        "Mean Energy Variability"
    )

    plt.ylabel(
        "Context-Dependence Score"
    )

    plt.title(
        "Energy Variability vs Context Dependence"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "figures/"
        "pilot_energy_variability_vs_context.png",
        dpi=300,
    )

    plt.show()

    # -------------------------
    # Spectral flux plot
    # -------------------------

    plt.figure(figsize=(7, 5))

    plt.scatter(
        combined["spectral_flux"],
        combined["Handcrafted"],
        label="Handcrafted",
    )

    plt.scatter(
        combined["spectral_flux"],
        combined["YAMNet"],
        label="YAMNet",
    )

    for category, row in combined.iterrows():

        plt.annotate(
            category,
            (
                row["spectral_flux"],
                row["Handcrafted"],
            ),
            fontsize=8,
        )

    plt.xlabel(
        "Mean Spectral Flux"
    )

    plt.ylabel(
        "Context-Dependence Score"
    )

    plt.title(
        "Spectral Flux vs Context Dependence"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        "figures/"
        "pilot_spectral_flux_vs_context.png",
        dpi=300,
    )

    plt.show()

    print(
        "\nSaved results to:",
        OUTPUT_CSV,
    )