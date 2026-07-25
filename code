"""
Healthcare Appointment No-Show Prediction & Access Optimization
===============================================================
A complete ML pipeline for predicting patient no-shows and simulating
smart overbooking strategies to improve clinic access.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Install with: pip install xgboost")

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Install with: pip install shap")

# Set visual style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

DATA_PATH = "healthcare_appointment_no_show_dataset.xlsx"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# =============================================================================
# 2. LOAD DATA
# =============================================================================

def load_data(path):
    """Load the synthetic appointment dataset."""
    print(f"Loading data from: {path}")
    df = pd.read_excel(path, sheet_name='Appointments')
    print(f"Loaded {len(df):,} records with {len(df.columns)} columns.\n")
    return df


# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def run_eda(df):
    """Generate EDA plots and summary statistics."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Basic stats
    print(f"\nNo-Show Rate: {df['No_Show'].mean()*100:.1f}%")
    print(f"\nClass Distribution:\n{df['No_Show'].value_counts()}\n")
    
    # Create output directory for plots
    import os
    os.makedirs("output_plots", exist_ok=True)
    
    # Plot 1: No-Show by Specialty
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    specialty_ns = df.groupby('Specialty')['No_Show'].mean().sort_values(ascending=False)
    specialty_ns.plot(kind='bar', ax=axes[0,0], color='steelblue')
    axes[0,0].set_title('No-Show Rate by Specialty', fontweight='bold')
    axes[0,0].set_ylabel('No-Show Rate')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: No-Show by Insurance Type
    insurance_ns = df.groupby('Insurance_Type')['No_Show'].mean().sort_values(ascending=False)
    insurance_ns.plot(kind='bar', ax=axes[0,1], color='coral')
    axes[0,1].set_title('No-Show Rate by Insurance Type', fontweight='bold')
    axes[0,1].set_ylabel('No-Show Rate')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Distance vs No-Show
    df.boxplot(column='Distance_from_Clinic_km', by='No_Show', ax=axes[1,0])
    axes[1,0].set_title('Distance Distribution by No-Show Status')
    axes[1,0].set_xlabel('No_Show (0=Showed, 1=No-Show)')
    
    # Plot 4: Lead Time vs No-Show
    df.boxplot(column='Appointment_Lead_Days', by='No_Show', ax=axes[1,1])
    axes[1,1].set_title('Lead Time Distribution by No-Show Status')
    axes[1,1].set_xlabel('No_Show (0=Showed, 1=No-Show)')
    
    plt.tight_layout()
    plt.savefig("output_plots/eda_overview.png", dpi=150, bbox_inches='tight')
    print("Saved: output_plots/eda_overview.png")
    plt.show()
    
    # Plot 5: Correlation heatmap (numeric features)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='RdBu_r', center=0, 
                fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig("output_plots/correlation_heatmap.png", dpi=150, bbox_inches='tight')
    print("Saved: output_plots/correlation_heatmap.png")
    plt.show()
    
    # Plot 6: Previous No-Shows impact
    prev_ns = df.groupby('Previous_No_Shows')['No_Show'].mean()
    plt.figure(figsize=(10, 5))
    prev_ns.plot(kind='bar', color='teal')
    plt.title('No-Show Rate by Previous No-Show Count', fontweight='bold')
    plt.xlabel('Previous No-Shows')
    plt.ylabel('No-Show Rate')
    plt.axhline(y=df['No_Show'].mean(), color='red', linestyle='--', 
                label=f'Overall Avg ({df["No_Show"].mean():.2f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig("output_plots/previous_noshows.png", dpi=150, bbox_inches='tight')
    print("Saved: output_plots/previous_noshows.png")
    plt.show()
    
    print("\nEDA complete.\n")


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """Create additional features from raw data."""
    df = df.copy()
    
    # Extract month and day-of-year from appointment date
    df['Appointment_Month'] = df['Appointment_Date'].dt.month
    df['Is_Weekend'] = (df['Appointment_Day'].isin(['Saturday', 'Sunday'])).astype(int)
    
    # Categorize lead time
    df['Lead_Time_Category'] = pd.cut(
        df['Appointment_Lead_Days'],
        bins=[-1, 0, 3, 7, 14, 30, 100],
        labels=['Same Day', '1-3 Days', '4-7 Days', '1-2 Weeks', '2-4 Weeks', '1+ Month']
    )
    
    # Distance category
    df['Distance_Category'] = pd.cut(
        df['Distance_from_Clinic_km'],
        bins=[0, 2, 5, 10, 20, 100],
        labels['<2km', '2-5km', '5-10km', '10-20km', '20km+']
    )
    
    # Age group
    df['Age_Group'] = pd.cut(
        df['Age'],
        bins=[0, 25, 40, 60, 100],
        labels=['18-25', '26-40', '41-60', '60+']
    )
    
    # Interaction: High-risk profile
    df['High_Risk_Profile'] = (
        (df['Previous_No_Shows'] >= 2).astype(int) +
        (df['Distance_from_Clinic_km'] > 15).astype(int) +
        (df['Appointment_Lead_Days'] > 20).astype(int) +
        (df['SMS_Reminder_Received'] == 0).astype(int)
    )
    
    # Drop original date column (not useful for tree models)
    df = df.drop(columns=['Appointment_Date'])
    
    print(f"Feature engineering complete. Total features: {df.shape[1]-1}")
    return df


# =============================================================================
# 5. PREPROCESSING PIPELINE
# =============================================================================

def build_preprocessor(df, target_col='No_Show'):
    """Build sklearn preprocessing pipeline."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify column types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove Patient_ID from features (just an identifier)
    if 'Patient_ID' in numeric_features:
        numeric_features.remove('Patient_ID')
        X = X.drop(columns=['Patient_ID'])
    
    print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor, numeric_features, categorical_features


# =============================================================================
# 6. MODEL TRAINING & EVALUATION
# =============================================================================

def train_and_evaluate(X, y, preprocessor):
    """Train multiple models and compare performance."""
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train):,} | Test set: {len(X_test):,}")
    print(f"Train no-show rate: {y_train.mean()*100:.1f}%")
    print(f"Test no-show rate: {y_test.mean()*100:.1f}%")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        )
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            random_state=RANDOM_STATE, eval_metric='logloss'
        )
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        print(f"\n{'-'*40}")
        print(f"Training: {name}")
        print(f"{'-'*40}")
        
        # Build pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc')
        print(f"CV ROC-AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std()*2:.4f})")
        
        # Fit on full training set
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        }
        
        results[name] = metrics
        fitted_models[name] = pipeline
        
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:\n  {cm}")
    
    # Results comparison
    results_df = pd.DataFrame(results).T
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(results_df.round(4).to_string())
    
    # Plot ROC curves
    plt.figure(figsize=(10, 7))
    for name, pipeline in fitted_models.items():
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison', fontweight='bold', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output_plots/roc_curves.png", dpi=150, bbox_inches='tight')
    print("\nSaved: output_plots/roc_curves.png")
    plt.show()
    
    return fitted_models, X_test, y_test, results_df


# =============================================================================
# 7. SHAP EXPLAINABILITY
# =============================================================================

def run_shap_analysis(fitted_models, X_test, model_name='Random Forest'):
    """Generate SHAP summary plots for model interpretability."""
    if not SHAP_AVAILABLE:
        print("\nSHAP not available. Skipping explainability analysis.")
        return
    
    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)
    
    pipeline = fitted_models[model_name]
    
    # Get preprocessor and classifier from pipeline
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    # Transform test data
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names
    feature_names = []
    # Numeric feature names
    num_features = preprocessor.transformers_[0][2]
    feature_names.extend(num_features)
    # Categorical feature names
    cat_features = preprocessor.transformers_[1][2]
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(cat_features)
    feature_names.extend(cat_feature_names)
    
    # Create DataFrame with processed features
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    
    # TreeExplainer for tree-based models
    if model_name in ['Random Forest', 'XGBoost']:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_test_df)
        
        # For binary classification, TreeExplainer may return list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (No-Show)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_df, show=False, max_display=15)
        plt.title(f'SHAP Feature Importance — {model_name}', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"output_plots/shap_summary_{model_name.replace(' ', '_').lower()}.png", 
                    dpi=150, bbox_inches='tight')
        print(f"Saved: output_plots/shap_summary_{model_name.replace(' ', '_').lower()}.png")
        plt.show()
        
        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False, max_display=15)
        plt.title(f'SHAP Mean Impact — {model_name}', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"output_plots/shap_bar_{model_name.replace(' ', '_').lower()}.png", 
                    dpi=150, bbox_inches='tight')
        print(f"Saved: output_plots/shap_bar_{model_name.replace(' ', '_').lower()}.png")
        plt.show()
        
        print("\nTop 5 drivers of no-shows (by mean |SHAP value|):")
        mean_shap = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        print(mean_shap.head(5).to_string(index=False))
    else:
        print("SHAP TreeExplainer requires tree-based model. Use Random Forest or XGBoost.")


# =============================================================================
# 8. OVERBOOKING OPTIMIZATION SIMULATION
# =============================================================================

def simulate_overbooking(fitted_models, X_test, y_test, model_name='Random Forest'):
    """
    Simulate smart overbooking strategy:
    - Predict no-show probability for each appointment
    - Overbook slots with lowest predicted risk
    - Calculate access improvement vs. baseline
    """
    print("\n" + "=" * 60)
    print("SMART OVERBOOKING SIMULATION")
    print("=" * 60)
    
    pipeline = fitted_models[model_name]
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    test_results = X_test.copy()
    test_results['actual_no_show'] = y_test.values
    test_results['predicted_prob'] = y_proba
    test_results['predicted_no_show'] = (y_proba >= 0.5).astype(int)
    
    # Sort by predicted probability (ascending = most likely to show up = safest to overbook)
    test_results = test_results.sort_values('predicted_prob').reset_index(drop=True)
    
    print(f"\nBaseline (no overbooking):")
    baseline_noshows = test_results['actual_no_show'].sum()
    baseline_shows = len(test_results) - baseline_noshows
    print(f"  Total appointments: {len(test_results):,}")
    print(f"  Expected no-shows: {baseline_noshows:,}")
    print(f"  Expected patients who show: {baseline_shows:,}")
    
    # Simulation parameters
    overbooking_rates = [0.05, 0.10, 0.15, 0.20, 0.25]
    simulation_results = []
    
    for rate in overbooking_rates:
        n_overbook = int(len(test_results) * rate)
        
        # Overbook the N appointments with LOWEST predicted no-show probability
        # (i.e., the ones most likely to actually show up)
        overbooked = test_results.copy()
        overbooked['overbooked'] = 0
        overbooked.loc[:n_overbook-1, 'overbooked'] = 1
        
        # Calculate outcomes
        # Original slots: some no-show (wasted), some show
        # Overbooked slots: we add an extra patient. If original shows, we have a conflict.
        # Simplified model: overbooked slot fills an empty slot if original no-shows
        
        original_no_shows = overbooked['actual_no_show'].sum()
        original_shows = len(overbooked) - original_no_shows
        
        # Overbooked patients fill empty slots created by no-shows
        # But if we overbook too many, we create conflicts when original patient shows
        filled_slots = min(n_overbook, original_no_shows)
        conflicts = max(0, n_overbook - original_no_shows)
        
        # Net gain: filled_slots are patients who got access they wouldn't have had
        # Conflicts are double-bookings when both show up
        net_access_gain = filled_slots
        utilization_rate = (original_shows + filled_slots) / len(overbooked)
        
        simulation_results.append({
            'Overbooking_Rate': f"{rate*100:.0f}%",
            'Slots_Overbooked': n_overbook,
            'Empty_Slots_Filled': filled_slots,
            'Double_Booking_Conflicts': conflicts,
            'Net_Access_Gain': net_access_gain,
            'Clinic_Utilization': f"{utilization_rate*100:.1f}%"
        })
    
    sim_df = pd.DataFrame(simulation_results)
    print(f"\n{'='*60}")
    print("OVERBOOKING SIMULATION RESULTS")
    print(f"{'='*60}")
    print(sim_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x_pos = range(len(sim_df))
    axes[0].bar(x_pos, sim_df['Empty_Slots_Filled'], color='seagreen', alpha=0.8, label='Slots Filled')
    axes[0].bar(x_pos, sim_df['Double_Booking_Conflicts'], bottom=sim_df['Empty_Slots_Filled'],
                color='crimson', alpha=0.8, label='Conflicts')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(sim_df['Overbooking_Rate'])
    axes[0].set_xlabel('Overbooking Rate')
    axes[0].set_ylabel('Number of Appointments')
    axes[0].set_title('Smart Overbooking Outcomes', fontweight='bold')
    axes[0].legend()
    
    utilization = [float(x.strip('%')) for x in sim_df['Clinic_Utilization']]
    axes[1].plot(x_pos, utilization, marker='o', linewidth=2, markersize=8, color='steelblue')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(sim_df['Overbooking_Rate'])
    axes[1].set_xlabel('Overbooking Rate')
    axes[1].set_ylabel('Clinic Utilization (%)')
    axes[1].set_title('Clinic Utilization Improvement', fontweight='bold')
    axes[1].axhline(y=(baseline_shows/len(test_results))*100, color='red', linestyle='--',
                    label=f'Baseline ({(baseline_shows/len(test_results))*100:.1f}%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("output_plots/overbooking_simulation.png", dpi=150, bbox_inches='tight')
    print("\nSaved: output_plots/overbooking_simulation.png")
    plt.show()
    
    # Business impact summary
    print(f"\n{'='*60}")
    print("BUSINESS IMPACT SUMMARY")
    print(f"{'='*60}")
    best = sim_df.iloc[2]  # 15% overbooking
    print(f"At a 15% smart overbooking rate:")
    print(f"  • {best['Empty_Slots_Filled']} additional patients get access to care")
    print(f"  • Only {best['Double_Booking_Conflicts']} scheduling conflicts expected")
    print(f"  • Clinic utilization improves from {(baseline_shows/len(test_results))*100:.1f}% to {best['Clinic_Utilization']}")
    
    return sim_df


# =============================================================================
# 9. PREDICTION FUNCTION (For Production Use)
# =============================================================================

def predict_no_show(pipeline, patient_data_dict):
    """
    Predict no-show probability for a single patient.
    
    Args:
        pipeline: Trained sklearn pipeline
        patient_data_dict: Dictionary with patient features
    
    Returns:
        Dictionary with prediction and probability
    """
    input_df = pd.DataFrame([patient_data_dict])
    proba = pipeline.predict_proba(input_df)[0, 1]
    prediction = int(proba >= 0.5)
    
    risk_level = 'Low' if proba < 0.3 else ('Medium' if proba < 0.6 else 'High')
    
    return {
        'no_show_probability': round(proba, 4),
        'predicted_no_show': prediction,
        'risk_level': risk_level,
        'recommendation': 'Send SMS reminder + call 24h before' if proba > 0.4 else 'Standard reminder'
    }


# =============================================================================
# 10. MAIN EXECUTION
# =============================================================================

def main():
    """Run the complete healthcare no-show prediction pipeline."""
    start_time = datetime.now()
    print(f"\n{'#'*60}")
    print(f"# HEALTHCARE NO-SHOW PREDICTION PIPELINE")
    print(f"# Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")
    
    # Step 1: Load
    df = load_data(DATA_PATH)
    
    # Step 2: EDA
    run_eda(df)
    
    # Step 3: Feature Engineering
    df = engineer_features(df)
    
    # Step 4: Preprocessing
    X, y, preprocessor, num_features, cat_features = build_preprocessor(df)
    
    # Step 5: Train & Evaluate
    fitted_models, X_test, y_test, results_df = train_and_evaluate(X, y, preprocessor)
    
    # Step 6: SHAP Explainability
    run_shap_analysis(fitted_models, X_test, model_name='Random Forest')
    
    # Step 7: Overbooking Simulation
    simulate_overbooking(fitted_models, X_test, y_test, model_name='Random Forest')
    
    # Step 8: Example single prediction
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTION")
    print("=" * 60)
    sample_patient = {
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
        'Appointment_Month': 6,
        'Is_Weekend': 0,
        'Lead_Time_Category': '2-4 Weeks',
        'Distance_Category': '10-20km',
        'Age_Group': '26-40',
        'High_Risk_Profile': 3
    }
    result = predict_no_show(fitted_models['Random Forest'], sample_patient)
    print(f"\nSample Patient Prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # Save results
    results_df.to_csv("output_plots/model_comparison.csv")
    print(f"\nSaved: output_plots/model_comparison.csv")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n{'#'*60}")
    print(f"# PIPELINE COMPLETE")
    print(f"# Duration: {duration:.1f} seconds")
    print(f"# All outputs saved to: output_plots/")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
