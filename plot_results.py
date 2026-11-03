import pandas as pd
import matplotlib.pyplot as plt

# Load experiment results
df = pd.read_csv("experiment_results.csv")

# Remove the average row for the comparison chart
df = df[df["Test_Recording"] != "AVERAGE"]

# Create the chart
plt.figure(figsize=(9, 5))

plt.bar(
    df["Test_Recording"],
    df["Accuracy"],
    label="Accuracy"
)

plt.axhline(
    df["Accuracy"].mean(),
    linestyle="--",
    label=f"Average: {df['Accuracy'].mean():.1f}%"
)

plt.title("NeuroScope: EEG Classification Performance")
plt.xlabel("Held-out EEG Recording")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

plt.savefig("model_evaluation.png", dpi=300)
plt.show()

print("Chart saved as model_evaluation.png")