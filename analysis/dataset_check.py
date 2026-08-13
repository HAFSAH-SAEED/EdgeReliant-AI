
import pandas as pd
import matplotlib.pyplot as plt


# Load the simulated dataset
df = pd.read_csv("data/sensor_data.csv")


# --------------------------------------------------
# Basic dataset information
# --------------------------------------------------

print("DATASET SHAPE")
print(df.shape)


print("\nCOLUMN NAMES")
print(df.columns.tolist())


# --------------------------------------------------
# Missing values
# --------------------------------------------------

print("\nMISSING VALUES")
print(df.isnull().sum())


# --------------------------------------------------
# Duplicate rows
# --------------------------------------------------

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())


# --------------------------------------------------
# Fault distribution
# --------------------------------------------------

print("\nFAULT DISTRIBUTION")
print(df["fault"].value_counts())


# --------------------------------------------------
# Numerical statistics
# --------------------------------------------------

print("\nNUMERICAL STATISTICS")
print(df.describe())


# --------------------------------------------------
# Sensor ranges
# --------------------------------------------------

print("\nSENSOR RANGES")

print(
    "Temperature:",
    df["temperature"].min(),
    "to",
    df["temperature"].max()
)

print(
    "Vibration:",
    df["vibration"].min(),
    "to",
    df["vibration"].max()
)

print(
    "Pressure:",
    df["pressure"].min(),
    "to",
    df["pressure"].max()
)

print(
    "Voltage:",
    df["voltage"].min(),
    "to",
    df["voltage"].max()
)

print(
    "Current:",
    df["current"].min(),
    "to",
    df["current"].max()
)


# --------------------------------------------------
# Final validation
# --------------------------------------------------

if df.isnull().sum().sum() == 0:
    print("\nPASS: No missing values.")

if df.duplicated().sum() == 0:
    print("PASS: No duplicate rows.")

if df["fault"].nunique() == 5:
    print("PASS: All five fault classes are present.")
    # --------------------------------------------------
# Sensor correlation analysis
# --------------------------------------------------

print("\nSENSOR CORRELATION MATRIX")

sensor_columns = [
    "temperature",
    "vibration",
    "pressure",
    "voltage",
    "current"
]

correlation_matrix = df[sensor_columns].corr()

print(correlation_matrix)
plt.figure(figsize=(8, 6))

plt.imshow(correlation_matrix, cmap="coolwarm")

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(sensor_columns)),
    sensor_columns,
    rotation=45
)

plt.yticks(
    range(len(sensor_columns)),
    sensor_columns
)

plt.title("Sensor Correlation Matrix")

plt.tight_layout()
plt.show()
# --------------------------------------------------
# Fault-specific correlation analysis
# --------------------------------------------------

print("\nFAULT-SPECIFIC CORRELATIONS")

for fault_name, group in df.groupby("fault"):

    print("\n----------------------------------------")
    print(fault_name)
    print("----------------------------------------")

    print(
        group[sensor_columns].corr().round(2)
    )