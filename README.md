
# EdgeReliant-AI

### From Sensor Data to Engineering Decisions

Built by **Hafsah Saeed**, Computer Engineering student @ NUST, exploring where embedded systems meet machine learning.

Most ML fault-detection projects stop at *"Fault detected."*

**EdgeReliant-AI doesn't.**

    Sensor Data → Validation → ML Classification → Confidence
         → Severity → Recommended Action → Verification

A predictive-maintenance prototype that turns multi-sensor readings into an engineering response — not just a label.

---

## What It Detects

Five sensor features:

**Temperature · Vibration · Pressure · Voltage · Current**

Five machine conditions:

`Normal` · `Overheating` · `High Vibration` · `Pressure Anomaly` · `Electrical Instability`

The model is trained on a validated, balanced simulated dataset containing **5,000 observations**, with **1,000 samples per class**, **zero missing values**, and **zero duplicate rows**.

---

## Results

Two machine-learning models were evaluated using the same train/test split:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| **Logistic Regression** | **90.60%** | **90.89%** |
| Random Forest | 89.50% | 89.74% |

Logistic Regression performed better across **accuracy, macro precision, macro recall, and macro F1**, so it was selected as the final model.

The complete preprocessing + classification pipeline is saved as:

`models/final_model.pkl`

### Multi-Fault Test

Representative test cases were created for all five fault classes:

| Test Case | Prediction | Confidence |
|---|---|---:|
| Overheating | ✅ Overheating | 100.00% |
| High Vibration | ✅ High Vibration | 100.00% |
| Pressure Anomaly | ✅ Pressure Anomaly | 100.00% |
| Electrical Instability | ✅ Electrical Instability | 100.00% |
| Normal | ✅ Normal | 63.87% |

All five representative test cases were classified into their intended fault categories.

The lower confidence for the Normal case is intentionally preserved rather than artificially adjusted.

---

## Not Just a Label — A Response

For example, given:

    Temperature: 75
    Vibration: 0.20
    Pressure: 101
    Voltage: 12
    Current: 2.4

EdgeReliant-AI produces:

    Fault: Overheating
    Confidence: 100.00%
    Severity: HIGH

    Action:
    Inspect the cooling system and reduce machine load if necessary.

    Verify:
    Re-check temperature after corrective action and confirm
    that it returns toward the normal operating range.

The confidence value represents the model's estimated probability for the predicted class — not a guarantee of real-world correctness.

---

## Under the Hood

The project includes:

- Dataset validation and quality checks
- Exploratory data analysis
- Sensor correlation analysis
- Fault-specific correlation analysis
- Logistic Regression
- Random Forest
- Model comparison
- Confusion matrix evaluation
- Feature importance analysis
- Multi-fault testing
- Saved model pipeline
- Fault severity and response rules

The strongest overall sensor relationship observed was:

**Voltage ↔ Current: r ≈ -0.67**

Random Forest feature importance ranked:

**Vibration → Pressure → Temperature → Voltage → Current**

as the top-to-bottom contributors in that model.

---

## System Architecture

    ┌─────────────────────┐
    │   Sensor Readings   │
    └──────────┬──────────┘
               ↓
    ┌─────────────────────┐
    │ Data Validation &   │
    │ Sensor Analysis     │
    └──────────┬──────────┘
               ↓
    ┌─────────────────────┐
    │   ML Classification │
    └──────────┬──────────┘
               ↓
    ┌─────────────────────┐
    │ Fault + Confidence  │
    └──────────┬──────────┘
               ↓
    ┌─────────────────────┐
    │ Severity Assessment │
    └──────────┬──────────┘
               ↓
    ┌─────────────────────┐
    │ Recommended Action  │
    └──────────┬──────────┘
               ↓
    ┌─────────────────────┐
    │     Verification    │
    └─────────────────────┘

---

## Project Structure

    EdgeReliant-AI/
    │
    ├── analysis/
    │   ├── dataset_check.py
    │   ├── logistic_confusion_matrix.py
    │   ├── ml_model.py
    │   ├── model_comparison.py
    │   ├── model_performance_chart.py
    │   ├── test_predictions.py
    │   └── train_model.py
    │
    ├── data/
    │   └── sensor_data.csv
    │
    ├── models/
    │   └── final_model.pkl
    │
    ├── results/
    │   ├── evaluation metrics
    │   ├── confusion matrices
    │   ├── model comparison
    │   └── multi-fault results
    │
    ├── src/
    │   ├── fault_rules.py
    │   ├── final_model.py
    │   ├── predict_fault.py
    │   └── sensor_simulator.py
    │
    ├── .gitignore
    ├── README.md
    └── requirements.txt

---

## Quick Start

Install dependencies:

    pip install -r requirements.txt

Generate sensor data:

    python src/sensor_simulator.py

Validate the dataset:

    python analysis/dataset_check.py

Train the final model:

    python src/final_model.py

Run fault prediction:

    python src/predict_fault.py

Run the multi-fault test suite:

    python analysis/test_predictions.py

Compare models:

    python analysis/model_comparison.py

---

## Technology Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Joblib`

---

## Roadmap

**Real sensor integration**  
**Edge deployment**  
**Time-series fault prediction**  
**Anomaly detection for unseen faults**  
**Explainable AI**  
**Computer-vision-based inspection**

---

## Why I Built This

I'm interested in the intersection of **hardware and intelligence**, not ML as a standalone black box, but ML as one link in a real engineering decision chain.

Predictive maintenance felt like the right sandbox: sensors, signal relationships, classification, and the part many basic ML projects stop short of, turning a prediction into an instruction an engineer can act on.

This is a prototype, and I know it.

The data is simulated. There are no physical sensors yet. The model has not been validated on an industrial machine or deployed to an edge device.

But the architecture is intentionally built to move in that direction:

**Sensors → Edge ML → Diagnosis → Engineering Action**

---

## Project Status

**Functional Machine-Learning Prototype**

Built around one idea:

> **Good ML doesn't just answer "what happened." It helps answer "what do I do next?"**