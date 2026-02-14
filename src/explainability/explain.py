import shap
import pickle
import pandas as pd
import numpy as np

def generate_explanation(patient_data):
    """Step 1: SHAP explainability for Multi-class (Phase 4)"""
    with open('models/risk_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    # 1. Prepare data
    df = pd.DataFrame([patient_data])
    df['arrival_mode'] = encoder.transform(df['arrival_mode'])
    
    # 2. Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)
    
    # 3. Get prediction to know which class to explain
    prediction = int(model.predict(df)[0])
    
    # Handle SHAP multi-class output shape variations
    # If shap_values is a list, we take the index of the prediction
    if isinstance(shap_values, list):
        current_shap_values = shap_values[prediction][0]
    else:
        # If it's a 3D array (samples, features, classes)
        current_shap_values = shap_values[0, :, prediction]
    
    # 4. Map back to feature names
    feature_importance = pd.Series(current_shap_values, index=df.columns)
    
    # 5. Extract top 3 influencing factors
    top_factors = feature_importance.abs().sort_values(ascending=False).head(3)
    
    explanation_text = f"Triage Level {prediction} Decision Logic:\n"
    for feature, val in top_factors.items():
        # High positive SHAP means this feature pushed the level HIGHER
        influence = "Critical factor" if val > 0 else "Protective factor"
        explanation_text += f"- {feature}: {influence}\n"
        
    return explanation_text

if __name__ == "__main__":
    sample_patient = {
        'age': 75.0, 'heart_rate': 110.0, 'systolic_blood_pressure': 150.0,
        'oxygen_saturation': 93.0, 'body_temperature': 38.5, 'pain_level': 7,
        'chronic_disease_count': 3, 'previous_er_visits': 2, 'arrival_mode': 'ambulance'
    }
    print(generate_explanation(sample_patient))