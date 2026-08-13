import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# Make the simulation reproducible
np.random.seed(42)


# Simulation settings
samples_per_state = 1000
number_of_states = 5
total_samples = samples_per_state * number_of_states

start_time = datetime.now()

data = []


for i in range(total_samples):
    timestamp = start_time + timedelta(seconds=i)

    # --------------------------------------------------
    # Normal operating condition
    # --------------------------------------------------
    if i < samples_per_state:

        temperature = np.random.normal(45, 2)
        vibration = np.random.normal(0.20, 0.03)
        pressure = np.random.normal(101.3, 1.0)
        voltage = np.random.normal(12.0, 0.15)
        current = np.random.normal(1.8, 0.12)

        fault = "Normal"


    # --------------------------------------------------
    # Gradual overheating condition
    # --------------------------------------------------
    elif i < samples_per_state * 2:

        progress = (i - samples_per_state) / samples_per_state

        temperature = (
            45
            + 35 * progress
            + np.random.normal(0, 2)
        )

        vibration = np.random.normal(0.20, 0.04)
        pressure = np.random.normal(101.3, 1.0)
        voltage = np.random.normal(12.0, 0.15)

        # Current rises slightly as machine heats up
        current = (
            1.8
            + 0.7 * progress
            + np.random.normal(0, 0.12)
        )

        fault = "Overheating"


    # --------------------------------------------------
    # Gradual high vibration condition
    # --------------------------------------------------
    elif i < samples_per_state * 3:

        progress = (i - samples_per_state * 2) / samples_per_state

        temperature = np.random.normal(45, 2)

        vibration = (
            0.20
            + 1.50 * progress
            + np.random.normal(0, 0.08)
        )

        pressure = np.random.normal(101.3, 1.0)
        voltage = np.random.normal(12.0, 0.15)
        current = np.random.normal(1.8, 0.12)

        fault = "High Vibration"


    # --------------------------------------------------
    # Gradual pressure anomaly condition
    # --------------------------------------------------
    elif i < samples_per_state * 4:

        progress = (i - samples_per_state * 3) / samples_per_state

        temperature = np.random.normal(45, 2)
        vibration = np.random.normal(0.20, 0.04)

        pressure = (
            101.3
            - 20 * progress
            + np.random.normal(0, 1.0)
        )

        voltage = np.random.normal(12.0, 0.15)
        current = np.random.normal(1.8, 0.12)

        fault = "Pressure Anomaly"


    # --------------------------------------------------
    # Electrical instability condition
    # --------------------------------------------------
    else:

        progress = (i - samples_per_state * 4) / samples_per_state

        temperature = (
            45
            + 5 * progress
            + np.random.normal(0, 2)
        )

        vibration = np.random.normal(0.20, 0.05)
        pressure = np.random.normal(101.3, 1.0)

        # Voltage becomes increasingly unstable
        voltage = (
            12.0
            - 2.0 * progress
            + np.random.normal(0, 0.25)
        )

        # Current increases
        current = (
            1.8
            + 1.0 * progress
            + np.random.normal(0, 0.15)
        )

        fault = "Electrical Instability"


    data.append({
        "timestamp": timestamp,
        "temperature": round(temperature, 2),
        "vibration": round(max(vibration, 0), 3),
        "pressure": round(pressure, 2),
        "voltage": round(voltage, 2),
        "current": round(max(current, 0), 2),
        "fault": fault
    })


# Convert to DataFrame
df = pd.DataFrame(data)


# --------------------------------------------------
# Dataset information
# --------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nFault distribution:")
print(df["fault"].value_counts())

print("\nFirst five rows:")
print(df.head())


# Save dataset
df.to_csv("data/sensor_data.csv", index=False)

print("\nDataset saved to data/sensor_data.csv")


# --------------------------------------------------
# Separate fault states
# --------------------------------------------------

normal = df["fault"] == "Normal"
overheating = df["fault"] == "Overheating"
high_vibration = df["fault"] == "High Vibration"
pressure_anomaly = df["fault"] == "Pressure Anomaly"
electrical_instability = df["fault"] == "Electrical Instability"


# --------------------------------------------------
# Temperature graph
# --------------------------------------------------

plt.figure()

plt.scatter(
    df.index[normal],
    df.loc[normal, "temperature"],
    label="Normal",
    s=6
)

plt.scatter(
    df.index[overheating],
    df.loc[overheating, "temperature"],
    label="Overheating",
    s=6
)

plt.scatter(
    df.index[high_vibration],
    df.loc[high_vibration, "temperature"],
    label="High Vibration",
    s=6
)

plt.scatter(
    df.index[pressure_anomaly],
    df.loc[pressure_anomaly, "temperature"],
    label="Pressure Anomaly",
    s=6
)

plt.scatter(
    df.index[electrical_instability],
    df.loc[electrical_instability, "temperature"],
    label="Electrical Instability",
    s=6
)

plt.xlabel("Reading")
plt.ylabel("Temperature (°C)")
plt.title("Simulated Machine Temperature")
plt.legend()
plt.show()


# --------------------------------------------------
# Vibration graph
# --------------------------------------------------

plt.figure()

plt.scatter(
    df.index[normal],
    df.loc[normal, "vibration"],
    label="Normal",
    s=6
)

plt.scatter(
    df.index[overheating],
    df.loc[overheating, "vibration"],
    label="Overheating",
    s=6
)

plt.scatter(
    df.index[high_vibration],
    df.loc[high_vibration, "vibration"],
    label="High Vibration",
    s=6
)

plt.scatter(
    df.index[pressure_anomaly],
    df.loc[pressure_anomaly, "vibration"],
    label="Pressure Anomaly",
    s=6
)

plt.scatter(
    df.index[electrical_instability],
    df.loc[electrical_instability, "vibration"],
    label="Electrical Instability",
    s=6
)

plt.xlabel("Reading")
plt.ylabel("Vibration")
plt.title("Simulated Machine Vibration")
plt.legend()
plt.show()


# --------------------------------------------------
# Pressure graph
# --------------------------------------------------

plt.figure()

plt.scatter(
    df.index[normal],
    df.loc[normal, "pressure"],
    label="Normal",
    s=6
)

plt.scatter(
    df.index[overheating],
    df.loc[overheating, "pressure"],
    label="Overheating",
    s=6
)

plt.scatter(
    df.index[high_vibration],
    df.loc[high_vibration, "pressure"],
    label="High Vibration",
    s=6
)

plt.scatter(
    df.index[pressure_anomaly],
    df.loc[pressure_anomaly, "pressure"],
    label="Pressure Anomaly",
    s=6
)

plt.scatter(
    df.index[electrical_instability],
    df.loc[electrical_instability, "pressure"],
    label="Electrical Instability",
    s=6
)

plt.xlabel("Reading")
plt.ylabel("Pressure")
plt.title("Simulated Machine Pressure")
plt.legend()
plt.show()


# --------------------------------------------------
# Voltage graph
# --------------------------------------------------

plt.figure()

plt.scatter(
    df.index[normal],
    df.loc[normal, "voltage"],
    label="Normal",
    s=6
)

plt.scatter(
    df.index[overheating],
    df.loc[overheating, "voltage"],
    label="Overheating",
    s=6
)

plt.scatter(
    df.index[high_vibration],
    df.loc[high_vibration, "voltage"],
    label="High Vibration",
    s=6
)

plt.scatter(
    df.index[pressure_anomaly],
    df.loc[pressure_anomaly, "voltage"],
    label="Pressure Anomaly",
    s=6
)

plt.scatter(
    df.index[electrical_instability],
    df.loc[electrical_instability, "voltage"],
    label="Electrical Instability",
    s=6
)

plt.xlabel("Reading")
plt.ylabel("Voltage (V)")
plt.title("Simulated Machine Voltage")
plt.legend()
plt.show()


# --------------------------------------------------
# Current graph
# --------------------------------------------------

plt.figure()

plt.scatter(
    df.index[normal],
    df.loc[normal, "current"],
    label="Normal",
    s=6
)

plt.scatter(
    df.index[overheating],
    df.loc[overheating, "current"],
    label="Overheating",
    s=6
)

plt.scatter(
    df.index[high_vibration],
    df.loc[high_vibration, "current"],
    label="High Vibration",
    s=6
)

plt.scatter(
    df.index[pressure_anomaly],
    df.loc[pressure_anomaly, "current"],
    label="Pressure Anomaly",
    s=6
)

plt.scatter(
    df.index[electrical_instability],
    df.loc[electrical_instability, "current"],
    label="Electrical Instability",
    s=6
)

plt.xlabel("Reading")
plt.ylabel("Current (A)")
plt.title("Simulated Machine Current")
plt.legend()
plt.show()