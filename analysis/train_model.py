
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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
# Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("TRAINING MODEL...")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# Create Logistic Regression pipeline
# --------------------------------------------------

model = Pipeline([
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
# Train model
# --------------------------------------------------

model.fit(X_train, y_train)


print("MODEL TRAINED SUCCESSFULLY")


# --------------------------------------------------
# Save trained model
# --------------------------------------------------

joblib.dump(
    model,
    "trained_model.pkl"
)


print("MODEL SAVED")
print("File: trained_model.pkl")