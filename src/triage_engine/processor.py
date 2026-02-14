import pickle
import pandas as pd
import numpy as np

class TriageProcessor:
    def __init__(self):
        # Load the trained model and encoder
        with open('models/risk_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            self.encoder = pickle.load(f)
            
    def get_department(self, patient_data):
        """Step 2: Department recommendation logic (Phase 3)"""
        # Simple mapping based on primary vitals/risk
        if patient_data['oxygen_saturation'] < 92:
            return "Pulmonology / Respiratory"
        if patient_data['heart_rate'] > 120 or patient_data['systolic_blood_pressure'] > 160:
            return "Cardiology"
        if patient_data['pain_level'] >= 8:
            return "Emergency / Trauma"
        return "General Medicine"

    def apply_rules(self, patient_data):
        """Step 3: Clinical rule engine safety layer (Phase 3)"""
        # Hard rules to prevent dangerous misclassification
        if patient_data['oxygen_saturation'] < 90:
            return 3, "CRITICAL: Low Oxygen Saturation"
        if patient_data['systolic_blood_pressure'] > 190:
            return 3, "CRITICAL: Severe Hypertension"
        return None, None

    def process_patient(self, patient_data):
        # 1. Check Safety Rules
        rule_level, reason = self.apply_rules(patient_data)
        
        # 2. Get ML Prediction
        input_df = pd.DataFrame([patient_data])
        # Fix encoding for categorical data
        input_df['arrival_mode'] = self.encoder.transform(input_df['arrival_mode'])
        ml_level = self.model.predict(input_df)[0]
        
        # Final Decision (Rules override ML)
        final_level = rule_level if rule_level is not None else ml_level
        source = "Safety Rule" if rule_level is not None else "ML Model"
        
        return {
            "triage_level": int(final_level),
            "department": self.get_department(patient_data),
            "source": source,
            "reason": reason
        }