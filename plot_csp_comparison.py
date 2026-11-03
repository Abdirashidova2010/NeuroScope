import pandas as pd
import matplotlib.pyplot as plt

# Load experiment results
df = pd.read_csv("csp_component_results.csv")

# Calculate average accuracy for each CSP configuration
averages = df.groupby("CSP_components")["Accuracy"].mean()

# Create comparison chart
plt.figure(figsize=(10, 6))

for recording in df["Test_Recording"].unique():
    subset = df[df["Test_Recording"] == recording]

    plt.plot(
        subset["CSP_components"],
        subset["Accuracy"],
        marker="o",
        label=recording
    )

plt.plot(
    averages.index,
    averages.values,
    marker="s",
    linewidth=3,
    linestyle="--",
    label="Average accuracy"
)

plt.title("CSP Component Performance Across EEG Recordings")
plt.xlabel("Number of CSP Components")
plt.ylabel("Accuracy (%)")

plt.xticks([4, 6, 8])
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("csp_component_comparison.png", dpi=300)
plt.show()

print("Chart saved successfully!")
print("\nAverage accuracy:")
print(averages)