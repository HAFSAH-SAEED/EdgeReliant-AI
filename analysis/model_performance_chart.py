
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Load model comparison results
# --------------------------------------------------

df = pd.read_csv("model_comparison_results.csv")


# --------------------------------------------------
# Convert metrics to percentages
# --------------------------------------------------

metrics = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1"
]

for metric in metrics:
    df[metric] = df[metric] * 100


# --------------------------------------------------
# Create performance chart
# --------------------------------------------------

ax = df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("EdgeReliant-AI Model Performance Comparison")
plt.ylabel("Score (%)")
plt.xlabel("Model")
plt.ylim(0, 100)

plt.xticks(rotation=0)

plt.legend(
    title="Metric"
)

plt.tight_layout()


# --------------------------------------------------
# Save chart
# --------------------------------------------------

plt.savefig(
    "model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


print("\nMODEL PERFORMANCE CHART")
print("-----------------------------------")
print("Graph saved successfully.")
print("File: model_performance_comparison.png")


# --------------------------------------------------
# Show chart
# --------------------------------------------------

plt.show()