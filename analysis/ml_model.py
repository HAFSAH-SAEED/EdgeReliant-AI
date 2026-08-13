import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv("data/sensor_data.csv")


# --------------------------------------------------
# Select sensor features and target
# --------------------------------------------------

features = [
    "temperature",
    "vibration",
    "pressure",
    "voltage",
    "current"
]

X = df[features]
y = df["fault"]


# --------------------------------------------------
# Split dataset into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("TRAINING SAMPLES:", len(X_train))
print("TESTING SAMPLES:", len(X_test))


# --------------------------------------------------
# Create Random Forest classifier
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# --------------------------------------------------
# Train model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# Calculate accuracy
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY")
print(f"{accuracy:.4f}")


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\nCLASSIFICATION REPORT")

report = classification_report(
    y_test,
    y_pred
)

print(report)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

print("\nCONFUSION MATRIX")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print(cm)


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

print("\nFEATURE IMPORTANCE")

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print(importance)


# --------------------------------------------------
# Calculate evaluation metrics
# --------------------------------------------------

report_dict = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

macro_precision = report_dict["macro avg"]["precision"]
macro_recall = report_dict["macro avg"]["recall"]
macro_f1 = report_dict["macro avg"]["f1-score"]


# --------------------------------------------------
# Save evaluation metrics
# --------------------------------------------------

metrics_text = f"""
RANDOM FOREST BASELINE EVALUATION

Training Samples: {len(X_train)}
Testing Samples: {len(X_test)}

Accuracy: {accuracy:.4f}
Accuracy Percentage: {accuracy * 100:.2f}%

Macro Precision: {macro_precision:.4f}
Macro Recall: {macro_recall:.4f}
Macro F1-Score: {macro_f1:.4f}
"""


with open(
    "random_forest_metrics.txt",
    "w"
) as file:
    file.write(metrics_text)


print("\nEVALUATION METRICS SAVED")
print("random_forest_metrics.txt")


# --------------------------------------------------
# Confusion matrix visualization
# --------------------------------------------------

plt.figure(figsize=(9, 7))

plt.imshow(cm)

plt.colorbar(
    label="Number of Predictions"
)

plt.xticks(
    range(len(model.classes_)),
    model.classes_,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(model.classes_)),
    model.classes_
)


# Add numbers inside the matrix

for i in range(len(model.classes_)):
    for j in range(len(model.classes_)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.xlabel("Predicted Fault")
plt.ylabel("Actual Fault")

plt.title(
    f"Random Forest Confusion Matrix\n"
    f"Accuracy = {accuracy:.2%}"
)

plt.tight_layout()


# --------------------------------------------------
# Save confusion matrix figure
# --------------------------------------------------

plt.savefig(
    "confusion_matrix_random_forest.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()