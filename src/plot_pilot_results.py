import pandas as pd
import matplotlib.pyplot as plt


RESULT_PATH = "results/pilot_handcrafted_results.csv"
FIGURE_PATH = "figures/pilot_handcrafted_performance.png"


df = pd.read_csv(RESULT_PATH)

summary = (
    df.groupby("duration")[["accuracy", "macro_f1"]]
    .mean()
    .reset_index()
)

plt.figure(figsize=(7, 5))

plt.plot(
    summary["duration"],
    summary["accuracy"],
    marker="o",
    label="Accuracy",
)

plt.plot(
    summary["duration"],
    summary["macro_f1"],
    marker="o",
    label="Macro-F1",
)

plt.xlabel("Observation Duration (seconds)")
plt.ylabel("Score")
plt.title("Pilot Classification Performance vs Observation Duration")
plt.xticks([1, 2, 5])
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_PATH, dpi=300)
plt.show()

print("Saved figure to:", FIGURE_PATH)