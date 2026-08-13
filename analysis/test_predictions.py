
import pandas as pd
import joblib


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

model = joblib.load("trained_model.pkl")


# --------------------------------------------------
# Test scenarios
# --------------------------------------------------

test_cases = [
    {
        "name": "Normal",
        "temperature": 46,
        "vibration": 0.20,
        "pressure": 101,
        "voltage": 12,
        "current": 1.9
    },

    {
        "name": "Overheating",
        "temperature": 75,
        "vibration": 0.20,
        "pressure": 101,
        "voltage": 12,
        "current": 2.4
    },

    {
        "name": "High Vibration",
        "temperature": 46,
        "vibration": 1.50,
        "pressure": 101,
        "voltage": 12,
        "current": 1.9
    },

    {
        "name": "Pressure Anomaly",
        "temperature": 46,
        "vibration": 0.20,
        "pressure": 80,
        "voltage": 12,
        "current": 1.9
    },

    {
        "name": "Electrical Instability",
        "temperature": 46,
        "vibration": 0.20,
        "pressure": 101,
        "voltage": 9.5,
        "current": 2.8
    }
]


# --------------------------------------------------
# Run predictions
# --------------------------------------------------

print("\nEDGE RELIANT AI - MULTI-FAULT TEST")
print("=" * 60)

results = []

for case in test_cases:

    new_data = pd.DataFrame(
        [[
            case["temperature"],
            case["vibration"],
            case["pressure"],
            case["voltage"],
            case["current"]
        ]],
        columns=[
            "temperature",
            "vibration",
            "pressure",
            "voltage",
            "current"
        ]
    )

    prediction = model.predict(new_data)[0]

    probabilities = model.predict_proba(new_data)[0]

    confidence = max(probabilities) * 100

    results.append({
        "Test Case": case["name"],
        "Predicted Fault": prediction,
        "Confidence": round(confidence, 2)
    })


# --------------------------------------------------
# Display results
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\nRESULTS")
print("-" * 60)
print(results_df.to_string(index=False))


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df.to_csv(
    "multi_fault_test_results.csv",
    index=False
)

print("\nRESULTS SAVED")
print("multi_fault_test_results.csv")