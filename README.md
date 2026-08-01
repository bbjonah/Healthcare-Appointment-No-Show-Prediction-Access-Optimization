# 🩺 Healthcare Appointment No-Show Prediction & Access Optimization

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2E86C1?style=for-the-badge&logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-8E44AD?style=for-the-badge)
![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

> An end-to-end **machine learning pipeline** that predicts patient no-show probability across 18,000 synthetic appointment records, explains *why* a patient might not show up using **SHAP**, and runs a **smart overbooking simulation** that fills empty clinic slots while minimising double-booking conflicts turning missed appointments into improved healthcare access.

</div>

---

## 📌 Problem

Missed healthcare appointments cost the US healthcare system an estimated **$150 billion annually**, and leave millions of patients waiting weeks for care that empty, unfilled slots could otherwise have provided.

Clinics typically respond with uniform overbooking or blanket reminder campaigns, neither of which accounts for *who* is actually likely to miss their appointment. There is a need for a **data-driven, interpretable, and reproducible pipeline** that predicts individual no-show risk, explains the drivers behind that risk, and translates those predictions into a practical scheduling strategy that increases clinic utilisation without overwhelming staff with conflicts.

---

## 🎯 Objective

- Predict the **no-show probability** for each scheduled appointment using machine learning
- Compare multiple classification models under **5-fold cross-validation**
- Explain individual and global predictions using **SHAP** for clinical decision-making
- Identify the **top behavioural, logistical, and demographic drivers** of no-shows
- Simulate a **risk-based smart overbooking strategy** to quantify its impact on clinic utilisation
- Deliver a **production-ready prediction function** for single-patient risk scoring with actionable recommendations
- Translate model outputs into measurable **business and health-equity impact**

---

## 🗂️ Dataset

All data is a **fully reproducible, privacy-safe synthetic dataset** generated for this project — no real patient records are used.

| Parameter | Value |
|-----------|-------|
| Size | 18,000 appointment records |
| No-Show Rate | 37.1% |
| Format | `.xlsx` |

### Features

| Feature | Type | Description |
|---------|------|--------------|
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
| `No_Show` | Target | 1 = missed appointment, 0 = showed up |

---

## 🛠️ Tools & Technologies

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn — pipelines, preprocessing, cross-validation
- **Gradient Boosting:** XGBoost
- **Explainable AI:** SHAP — summary and bar plots, per-patient explanations
- **Visualisation:** Matplotlib, Seaborn
- **Model Selection:** 5-fold cross-validation across 3 candidate models

---

## ⚙️ Methodology / Project Workflow

1. **Data Loading:** Ingest the synthetic 18,000-record appointment dataset (`healthcare_appointment_no_show_dataset.xlsx`)
2. **Exploratory Data Analysis:** Generate EDA overview and correlation heatmap visualisations to understand distributions and relationships between features
3. **Feature Engineering:** Derive engineered features including `Appointment_Month`, `Is_Weekend`, `Lead_Time_Category`, `Distance_Category`, `Age_Group`, and a composite `High_Risk_Profile` score
4. **Model Training:** Train and compare **Logistic Regression, Random Forest, and XGBoost** classifiers using 5-fold cross-validation
5. **Model Evaluation:** Score each model on Accuracy, Precision, Recall, F1-Score, and ROC-AUC; generate ROC curve comparison plots
6. **Explainability:** Apply **SHAP** to the best-performing model to identify global feature importance and generate per-patient explanations
7. **Overbooking Simulation:** Use predicted no-show probabilities to selectively overbook the lowest-risk appointment slots at varying overbooking rates, measuring slots filled vs. scheduling conflicts
8. **Prediction Function:** Expose a `predict_no_show()` function that returns a probability, binary prediction, risk level, and actionable recommendation for a single patient
9. **Export & Reporting:** Save all visualisations and a model comparison summary to `output_plots/`

---

## 📊 Key Features

- ✅ **End-to-end ML pipeline:** data loading → EDA → feature engineering → modelling → evaluation → explainability, all from a single `main.py`
- ✅ **3-model comparison:** Logistic Regression, Random Forest, and XGBoost benchmarked under 5-fold cross-validation
- ✅ **SHAP explainability:** interpretable, clinically trustworthy predictions at both the global and individual-patient level
- ✅ **Smart overbooking simulation:** risk-based overbooking that fills empty slots while minimising double-booking conflicts
- ✅ **Production-ready prediction function:** single-patient risk scoring with an actionable recommendation (e.g. SMS reminder, follow-up call)
- ✅ **Fully synthetic, reproducible dataset:** privacy-safe and included with the project
- ✅ **Business-impact framing:** translates model performance into additional patients seen and equity gains for underserved, waitlisted patients

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.72 | ~0.65 | ~0.68 | ~0.66 | ~0.78 |
| Random Forest | ~0.76 | ~0.71 | ~0.74 | ~0.72 | **~0.84** |
| XGBoost | ~0.75 | ~0.70 | ~0.73 | ~0.71 | ~0.83 |

**Best Model:** Random Forest — highest ROC-AUC and the best balance of precision and recall.

---

## 🔍 SHAP Explainability

Understanding *why* a model predicts a no-show is critical for clinical trust. SHAP analysis reveals the top drivers of no-shows:

1. **Previous No-Shows** — past behaviour is the strongest predictor
2. **Appointment Lead Time** — longer waits correlate with higher risk
3. **Distance from Clinic** — patients farther away are less likely to attend
4. **SMS Reminder** — a strong protective factor that reduces no-show risk
5. **Insurance Type** — uninsured patients show higher no-show rates

---

## 💡 Smart Overbooking Simulation

Rather than uniform overbooking, the pipeline uses predicted no-show probabilities to selectively overbook the **lowest-risk** appointments.

| Overbooking Rate | Empty Slots Filled | Conflicts | Clinic Utilization |
|-------------------|--------------------|-----------|---------------------|
| 5% | ~280 | ~40 | 66.2% |
| 10% | ~520 | ~110 | 69.8% |
| 15% | ~720 | ~180 | 72.4% |
| 20% | ~890 | ~310 | 74.1% |

**Recommended Strategy:** 15% smart overbooking yields ~720 additional patient visits with manageable conflict levels.

---

## 📁 Project Structure

```
no-show-prediction/
├── main.py                                      # Complete ML pipeline
├── healthcare_appointment_no_show_dataset.xlsx  # Synthetic dataset
├── requirements.txt                             # Python dependencies
├── README.md                                    # This file
├── output_plots/                                # Generated visualisations
│   ├── eda_overview.png
│   ├── correlation_heatmap.png
│   ├── roc_curves.png
│   ├── shap_summary_random_forest.png
│   ├── shap_bar_random_forest.png
│   ├── overbooking_simulation.png
│   └── model_comparison.csv
└── LICENSE
```

---

## ▶️ How to Run

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/no-show-prediction.git
cd no-show-prediction
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python main.py
```

**What the pipeline produces automatically:**

| Output | Location |
|--------|----------|
| EDA visualisations | `output_plots/eda_overview.png`, `output_plots/correlation_heatmap.png` |
| ROC curve comparison | `output_plots/roc_curves.png` |
| SHAP summary & bar plots | `output_plots/shap_summary_random_forest.png`, `output_plots/shap_bar_random_forest.png` |
| Overbooking simulation results | `output_plots/overbooking_simulation.png` |
| Model comparison table | `output_plots/model_comparison.csv` |

### Dependencies

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
openpyxl
```

---

## 🧠 Sample Prediction

```python
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
```

---

## 🎯 Business Impact

| Metric | Value |
|--------|-------|
| No-Show Prediction Accuracy | 84% ROC-AUC |
| Additional Patients Seen (15% overbooking) | ~720 per 5,000 appointments |
| Intervention ROI | Targeted SMS/call campaigns for high-risk patients |
| Equity Angle | Filling empty slots improves access for underserved patients on waitlists |

---

## ⚠️ Limitations & Future Work

**Current Limitations:**
- The dataset is **fully synthetic**, so absolute performance figures are illustrative rather than a guarantee on real clinic data
- The overbooking simulation assumes **static risk scores** and does not model real-time schedule changes or cancellations
- No **time-series component** — the model treats each appointment independently rather than tracking patient behaviour trends over time
- SHAP explanations are generated for the **Random Forest model only**, not cross-validated across all three model types

**Future Improvements:**
- 🌲 Add **LightGBM and CatBoost** to the model comparison, along with neural network baselines
- 📈 Incorporate **time-series analysis** of no-show trends per patient and per clinic
- 🔗 Integrate with **real clinic scheduling systems** for live prediction and overbooking
- 📊 Build a **Streamlit dashboard** for interactive, real-time no-show predictions
- 🧪 Validate the overbooking strategy against **real-world pilot data**
- 📱 Expand reminder interventions beyond SMS to include app notifications and automated calls

---

## 📚 References

- Kaggle Medical Appointment No Shows Dataset *(inspiration)*
- SHAP: A Unified Approach to Interpreting Model Predictions
- The Cost of Missed Appointments in Healthcare

---

## 🤝 Contributing

Contributions are welcome! Open an issue or pull request for:

- Additional models (LightGBM, CatBoost, Neural Networks)
- Time-series analysis of no-show trends
- Integration with real clinic scheduling systems
- A Streamlit dashboard for interactive predictions

---

## 📄 License

This project is licensed under the **MIT License** — free to use, adapt, and build upon for research, education, and healthcare access analytics.
See the [LICENSE](LICENSE) file for full details.

---

<div align="center">

**Built with 💙 to improve healthcare access**

</div>
