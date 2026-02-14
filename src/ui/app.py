import streamlit as st
import sys
import os

# Add the root directory to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.triage_engine.processor import TriageProcessor
from src.explainability.explain import generate_explanation

# Initialize the Engine
processor = TriageProcessor()

st.set_page_config(page_title="AI Clinical Triage System", layout="wide")

st.title("🏥 AI Clinical Triage Dashboard")
st.markdown("---")

# Layout: 2 Columns (Input and Output)
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Patient Intake Form")
    with st.form("patient_form"):
        # Demographics
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        arrival_mode = st.selectbox("Arrival Mode", ["walk_in", "ambulance", "wheelchair"])
        
        # Vitals
        hr = st.slider("Heart Rate (BPM)", 40, 200, 80)
        sbp = st.slider("Systolic BP", 70, 220, 120)
        spo2 = st.slider("Oxygen Saturation (SpO2 %)", 70, 100, 98)
        temp = st.number_input("Temperature (°C)", 35.0, 42.0, 37.0)
        
        # Clinical Context
        pain = st.select_slider("Pain Level", options=list(range(1, 11)), value=5)
        chronic = st.number_input("Chronic Diseases Count", 0, 10, 0)
        prev_er = st.number_input("Previous ER Visits", 0, 20, 0)
        
        submit = st.form_submit_button("Run Triage Analysis")

if submit:
    # Prepare data for model
    patient_data = {
        'age': float(age), 'heart_rate': float(hr), 'systolic_blood_pressure': float(sbp),
        'oxygen_saturation': float(spo2), 'body_temperature': float(temp), 'pain_level': int(pain),
        'chronic_disease_count': int(chronic), 'previous_er_visits': int(prev_er), 
        'arrival_mode': arrival_mode
    }
    
    # Run Engine
    result = processor.process_patient(patient_data)
    explanation = generate_explanation(patient_data)
    
    with col2:
        st.header("Triage Result")
        
        # Risk Card
       # Risk Card
        color = {0: "green", 1: "blue", 2: "orange", 3: "red"}[result['triage_level']]
        st.markdown(f"""
            <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                <h1 style="color: white; margin: 0;">LEVEL {result['triage_level']}</h1>
                <h3 style="color: white; margin: 0;">Source: {result['source']}</h3>
            </div>
        """, unsafe_allow_html=True) # Changed from unsafe_allow_value to unsafe_allow_html
        
        st.subheader(f"Recommended Department: **{result['department']}**")
        
        # Explanation Section
        st.info(explanation)
        
        if result['reason']:
            st.warning(f"Note: {result['reason']}")