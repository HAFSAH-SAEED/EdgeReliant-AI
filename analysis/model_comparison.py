import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv("data/sensor_data.csv")


# --------------------------------------------------
# Features and target
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
# Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Define models
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# --------------------------------------------------
# Evaluate models
# --------------------------------------------------

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1
    })


# --------------------------------------------------
# Create comparison table
# --------------------------------------------------

results_df = pd.DataFrame(results)


print("\nEDGE RELIANT AI - MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# --------------------------------------------------
# Save comparison results
# --------------------------------------------------

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)


print("\nRESULTS SAVED")
print("model_comparison_results.csv")