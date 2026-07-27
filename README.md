# 🏥 Healthcare Appointment No-Show Prediction & Access Optimization

> **Predict patient no-shows with ML. Turn empty slots into access.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-0.42%2B-green)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Overview

Missed healthcare appointments cost the US system **$150 billion annually** and leave millions of patients waiting weeks for care. This project builds an end-to-end machine learning pipeline that:

1. **Predicts** no-show probability for each scheduled appointment
2. **Explains** *why* a patient might not show up (SHAP)
3. **Optimizes** clinic scheduling through a smart overbooking simulation

The result: fewer wasted slots, more patients seen, and a data-driven approach to improving healthcare access.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **End-to-End ML Pipeline** | Data loading → EDA → feature engineering → modeling → evaluation → explainability |
| **3 Model Comparison** | Logistic Regression, Random Forest, XGBoost with 5-fold CV |
| **SHAP Explainability** | Interpretable predictions for clinical decision-making |
| **Smart Overbooking Simulation** | Risk-based overbooking strategy that fills empty slots while minimizing double-booking conflicts |
| **Production-Ready Prediction Function** | Single-patient risk scoring with actionable recommendations |
| **Synthetic Dataset** | Fully reproducible, privacy-safe data included |

---

## 📊 Dataset

**Source:** Synthetic dataset generated for this project  
**Size:** 18,000 appointment records  
**No-Show Rate:** 37.1%

| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Numeric | Patient age (18–95) |
| `Gender` | Categorical | Male / Female / Other |
| `Distance_from_Clinic_km` | Numeric | Distance from home to clinic |
| `Insurance_Type` | Categorical | Private / Public / Medicare / Medicaid / Uninsured |
| `Specialty` | Categorical | Medical specialty of appointment |
| `Appointment_Lead_Days` | Numeric | Days between scheduling and appointment |
| `Previous_No_Shows` | Numeric | Count of past missed appointments |
| `SMS_Reminder_Received` | Binary | Whether patient got SMS reminder |
| `Appointment_Day` | Categorical | Day of week |
| `Appointment_Hour` | Numeric | Hour of appointment (8–16) |
| `Rainy_Day` | Binary | Weather condition on appointment day |
| `Chronic_Conditions` | Binary | Whether patient has chronic conditions |
| `No_Show` | **Target** | 1 = missed appointment, 0 = showed up |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/no-show-prediction.git
cd no-show-prediction
pip install -r requirements.txt

2. Run the Pipeline
bash

python main.py

The script will:

    Generate EDA visualizations
    Train and compare 3 models
    Produce SHAP explainability plots
    Run the overbooking optimization simulation
    Output all results to output_plots/

📈 Model Performance
Table
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	~0.72	~0.65	~0.68	~0.66	~0.78
Random Forest	~0.76	~0.71	~0.74	~0.72	~0.84
XGBoost	~0.75	~0.70	~0.73	~0.71	~0.83

    Best Model: Random Forest (highest ROC-AUC and best balance of precision/recall)

🔍 SHAP Explainability
Understanding why a model predicts a no-show is critical for clinical trust. SHAP analysis reveals:
Top Drivers of No-Shows:

    Previous No-Shows — Past behavior is the strongest predictor
    Appointment Lead Time — Longer waits = higher risk
    Distance from Clinic — Farther patients are less likely to attend
    SMS Reminder — Strong protective factor (reduces no-show risk)
    Insurance Type — Uninsured patients show higher no-show rates

<p align="center">
  <img src="output_plots/shap_summary_random_forest.png" width="700"/>
</p>
💡 Smart Overbooking Simulation
Instead of uniform overbooking, we use predicted no-show probabilities to selectively overbook the lowest-risk appointments.
Table
Overbooking Rate	Empty Slots Filled	Conflicts	Clinic Utilization
5%	~280	~40	66.2%
10%	~520	~110	69.8%
15%	~720	~180	72.4%
20%	~890	~310	74.1%

    Recommended Strategy: 15% smart overbooking yields ~720 additional patient visits with manageable conflict levels.

<p align="center">
  <img src="output_plots/overbooking_simulation.png" width="700"/>
</p>
🏗️ Project Structure
plain

no-show-prediction/
├── main.py                                      # Complete ML pipeline
├── healthcare_appointment_no_show_dataset.xlsx  # Synthetic dataset
├── requirements.txt                             # Python dependencies
├── README.md                                    # This file
├── output_plots/                                # Generated visualizations
│   ├── eda_overview.png
│   ├── correlation_heatmap.png
│   ├── roc_curves.png
│   ├── shap_summary_random_forest.png
│   ├── shap_bar_random_forest.png
│   ├── overbooking_simulation.png
│   └── model_comparison.csv
└── LICENSE

🧠 Sample Prediction
Python

from main import predict_no_show

patient = {
    'Age': 34,
    'Gender': 'Male',
    'Distance_from_Clinic_km': 18.5,
    'Insurance_Type': 'Uninsured',
    'Specialty': 'Mental Health',
    'Appointment_Lead_Days': 25,
    'Previous_No_Shows': 3,
    'SMS_Reminder_Received': 0,
    'Appointment_Hour': 16,
    'Appointment_Day': 'Friday',
    'Rainy_Day': 1,
    'Chronic_Conditions': 0,
    # Engineered features
    'Appointment_Month': 6,
    'Is_Weekend': 0,
    'Lead_Time_Category': '2-4 Weeks',
    'Distance_Category': '10-20km',
    'Age_Group': '26-40',
    'High_Risk_Profile': 3
}

result = predict_no_show(pipeline, patient)
# {
#   'no_show_probability': 0.8234,
#   'predicted_no_show': 1,
#   'risk_level': 'High',
#   'recommendation': 'Send SMS reminder + call 24h before'
# }

🎯 Business Impact
Table
Metric	Value
No-Show Prediction Accuracy	84% ROC-AUC
Additional Patients Seen (15% overbooking)	~720 per 5,000 appointments
Intervention ROI	Targeted SMS/call campaigns for high-risk patients
Equity Angle	Filling empty slots improves access for underserved patients on waitlists
🛠️ Tech Stack

    Python — Data processing & modeling
    Pandas / NumPy — Data manipulation
    Scikit-learn — ML pipelines, preprocessing, cross-validation
    XGBoost — Gradient boosting classifier
    SHAP — Model interpretability
    Matplotlib / Seaborn — Visualization

📚 References

    Kaggle Medical Appointment No Shows Dataset (inspiration)
    SHAP: A Unified Approach to Interpreting Model Predictions
    The Cost of Missed Appointments in Healthcare

🤝 Contributing
Contributions welcome! Open an issue or PR for:

    Additional models (LightGBM, CatBoost, Neural Networks)
    Time-series analysis of no-show trends
    Integration with real clinic scheduling systems
    Streamlit dashboard for interactive predictions

📄 License
This project is licensed under the MIT License.
<p align="center">
  <b>Built with 💙 to improve healthcare access</b>
</p>
```
