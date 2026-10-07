import pandas as pd
import matplotlib.pyplot as plt


HANDCRAFTED_PATH = "results/pilot_handcrafted_results.csv"
YAMNET_PATH = "results/pilot_yamnet_results.csv"
FIGURE_PATH = "figures/pilot_representation_comparison.png"


handcrafted = pd.read_csv(HANDCRAFTED_PATH)
yamnet = pd.read_csv(YAMNET_PATH)

handcrafted_summary = (
    handcrafted
    .groupby("duration")[["accuracy", "macro_f1"]]
    .mean()
    .reset_index()
)

yamnet_summary = (
    yamnet
    .groupby("duration")[["accuracy", "macro_f1"]]
    .mean()
    .reset_index()
)


plt.figure(figsize=(8, 5))

plt.plot(
    handcrafted_summary["duration"],
    handcrafted_summary["macro_f1"],
    marker="o",
    label="Handcrafted Features",
)

plt.plot(
    yamnet_summary["duration"],
    yamnet_summary["macro_f1"],
    marker="o",
    label="Frozen YAMNet Embeddings",
)

plt.xlabel("Observation Duration (seconds)")
plt.ylabel("Macro-F1")
plt.title(
    "Environmental Sound Classification vs Observation Duration"
)

plt.xticks([1, 2, 5])
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    FIGURE_PATH,
    dpi=300
)

plt.show()

print("Saved figure to:", FIGURE_PATH)