
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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
# Final Logistic Regression model
# --------------------------------------------------

final_model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# --------------------------------------------------
# Train final model
# --------------------------------------------------

final_model.fit(X_train, y_train)


# --------------------------------------------------
# Verify model performance
# --------------------------------------------------

predictions = final_model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print("\nEDGE RELIANT AI - FINAL MODEL")
print("=" * 50)

print("Model: Logistic Regression")
print("Features:", ", ".join(features))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# Save final model
# --------------------------------------------------

joblib.dump(
    final_model,
    "models/final_model.pkl"
)

print("\nFINAL MODEL SAVED")
print("File: models/final_model.pkl")