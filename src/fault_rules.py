
# --------------------------------------------------
# EdgeReliant-AI Fault Response Rules
# --------------------------------------------------

fault_rules = {

    "Normal": {
        "severity": "LOW",
        "action": "Continue normal operation and routine monitoring.",
        "verification": "Continue monitoring sensor readings for abnormal changes."
    },

    "Overheating": {
        "severity": "HIGH",
        "action": "Inspect the cooling system and reduce machine load if necessary.",
        "verification": "Check temperature again after corrective action and confirm that it returns toward the normal operating range."
    },

    "High Vibration": {
        "severity": "HIGH",
        "action": "Inspect mechanical components for imbalance, looseness, misalignment, or excessive vibration.",
        "verification": "Recheck vibration readings after corrective action and confirm that vibration decreases."
    },

    "Pressure Anomaly": {
        "severity": "MEDIUM",
        "action": "Inspect the pressure system, connections, and pressure sensor.",
        "verification": "Recheck pressure readings and confirm that they return toward the expected operating range."
    },

    "Electrical Instability": {
        "severity": "HIGH",
        "action": "Inspect the electrical supply, wiring, connections, voltage, and current readings.",
        "verification": "After corrective action, verify that voltage and current readings are stable."
    }
}