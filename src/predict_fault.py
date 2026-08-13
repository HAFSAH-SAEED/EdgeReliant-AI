
import pandas as pd
import joblib

from fault_rules import fault_rules


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("trained_model.pkl")


# --------------------------------------------------
# Get sensor readings
# --------------------------------------------------

print("\nEDGE RELIANT AI - FAULT PREDICTION")
print("-----------------------------------")

temperature = float(
    input("Enter temperature: ")
)

vibration = float(
    input("Enter vibration: ")
)

pressure = float(
    input("Enter pressure: ")
)

voltage = float(
    input("Enter voltage: ")
)

current = float(
    input("Enter current: ")
)


# --------------------------------------------------
# Create sensor data
# --------------------------------------------------

new_data = pd.DataFrame(
    [[
        temperature,
        vibration,
        pressure,
        voltage,
        current
    ]],
    columns=[
        "temperature",
        "vibration",
        "pressure",
        "voltage",
        "current"
    ]
)


# --------------------------------------------------
# Predict fault
# --------------------------------------------------

prediction = model.predict(new_data)

probabilities = model.predict_proba(new_data)[0]

predicted_fault = prediction[0]

confidence = max(probabilities) * 100


# --------------------------------------------------
# Get engineering response
# --------------------------------------------------

response = fault_rules[predicted_fault]

severity = response["severity"]
action = response["action"]
verification = response["verification"]


# --------------------------------------------------
# Display complete result
# --------------------------------------------------

print("\nPREDICTION")
print("-----------------------------------")

print(
    "Predicted Fault:",
    predicted_fault
)

print(
    f"Confidence: {confidence:.2f}%"
)

print(
    "Severity:",
    severity
)

print(
    "\nRecommended Action:"
)

print(
    action
)

print(
    "\nVerification:"
)

print(
    verification
)