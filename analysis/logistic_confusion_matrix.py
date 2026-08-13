
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
# Create Logistic Regression model
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


# --------------------------------------------------
# Make predictions
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# Create confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)


# --------------------------------------------------
# Display confusion matrix
# --------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(
    cmap="Blues",
    xticks_rotation=45
)

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.tight_layout()


# --------------------------------------------------
# Save figure
# --------------------------------------------------

plt.savefig(
    "logistic_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)


print("\nLOGISTIC REGRESSION CONFUSION MATRIX")
print("-----------------------------------")
print(cm)

print("\nGRAPH SAVED")
print("logistic_confusion_matrix.png")


# --------------------------------------------------
# Show graph
# --------------------------------------------------

plt.show()