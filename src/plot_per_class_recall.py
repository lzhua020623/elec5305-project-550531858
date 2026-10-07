import pandas as pd
import matplotlib.pyplot as plt


HANDCRAFTED_PATH = (
    "results/pilot_handcrafted_per_class_recall.csv"
)

YAMNET_PATH = (
    "results/pilot_yamnet_per_class_recall.csv"
)


def plot_per_class_recall(
    csv_path,
    title,
    output_path,
):
    df = pd.read_csv(csv_path)

    summary = (
        df.groupby(
            ["category", "duration"]
        )["recall"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    for category in sorted(
        summary["category"].unique()
    ):
        category_df = summary[
            summary["category"] == category
        ]

        plt.plot(
            category_df["duration"],
            category_df["recall"],
            marker="o",
            label=category,
        )

    plt.xlabel(
        "Observation Duration (seconds)"
    )
    plt.ylabel("Recall")
    plt.title(title)

    plt.xticks([1, 2, 5])
    plt.ylim(0, 1.05)

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.show()

    print(
        "Saved figure to:",
        output_path,
    )


plot_per_class_recall(
    HANDCRAFTED_PATH,
    "Per-Class Recall vs Observation Duration "
    "- Handcrafted Features",
    "figures/"
    "pilot_handcrafted_per_class_recall.png",
)

plot_per_class_recall(
    YAMNET_PATH,
    "Per-Class Recall vs Observation Duration "
    "- Frozen YAMNet Embeddings",
    "figures/"
    "pilot_yamnet_per_class_recall.png",
)