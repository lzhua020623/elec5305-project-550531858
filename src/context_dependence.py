import pandas as pd
import matplotlib.pyplot as plt


HANDCRAFTED_PATH = (
    "results/pilot_handcrafted_per_class_recall.csv"
)

YAMNET_PATH = (
    "results/pilot_yamnet_per_class_recall.csv"
)


def compute_context_dependence(csv_path):
    df = pd.read_csv(csv_path)

    summary = (
        df.groupby(
            ["category", "duration"]
        )["recall"]
        .mean()
        .unstack()
    )

    summary["context_dependence"] = (
        summary[5] - summary[1]
    )

    return summary


handcrafted = compute_context_dependence(
    HANDCRAFTED_PATH
)

yamnet = compute_context_dependence(
    YAMNET_PATH
)


comparison = pd.DataFrame({
    "Handcrafted": handcrafted[
        "context_dependence"
    ],
    "YAMNet": yamnet[
        "context_dependence"
    ],
})


print(
    "\nContext-dependence scores:"
)

print(comparison)


comparison.to_csv(
    "results/"
    "pilot_context_dependence.csv"
)


ax = comparison.plot(
    kind="bar",
    figsize=(9, 5),
)

ax.set_xlabel("Sound Category")
ax.set_ylabel(
    "Context-Dependence Score "
    "(Recall 5s - Recall 1s)"
)

ax.set_title(
    "Context Dependence by Sound Category"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "figures/"
    "pilot_context_dependence.png",
    dpi=300,
)

plt.show()