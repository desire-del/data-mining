import streamlit as st
import requests
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

# ----------------------------
# 🎯 Pydantic model: Base Inputs Only
# ----------------------------
class PatientInput(BaseModel):
    patient_nbr: str = Field(..., min_length=1)  # Identification unique du patient
    encounter_id: str = Field(..., min_length=1)  # Identification unique de l'admission
    admission_type_id: int = Field(..., ge=1, le=9)
    age: Literal[
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ]
    discharge_disposition_id: int = Field(..., ge=1)
    admission_source_id: int = Field(..., ge=1)
    time_in_hospital: int = Field(..., ge=1)
    num_lab_procedures: int = Field(..., ge=0)
    num_procedures: int = Field(..., ge=0)
    num_medications: int = Field(..., ge=0)
    number_outpatient: int = Field(..., ge=0)
    number_emergency: int = Field(..., ge=0)
    number_inpatient: int = Field(..., ge=0)
    number_diagnoses: int = Field(..., ge=0)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"]
    A1Cresult: Literal["None", "Norm", ">7", ">8"]
    gender: Literal["Male", "Female"]
    race: Literal["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"]
    change: Literal["Ch", "No"]
    diabetesMed: Literal["Yes", "No"]
    diag_1: str = Field(..., min_length=1)
    diag_2: str = Field(..., min_length=1)
    diag_3: str = Field(..., min_length=1)
    metformin: Literal["No", "Steady", "Up", "Down"]
    repaglinide: Literal["No", "Steady", "Up", "Down"]
    glimepiride: Literal["No", "Steady", "Up", "Down"]
    pioglitazone: Literal["No", "Steady", "Up", "Down"]
    rosiglitazone: Literal["No", "Steady", "Up", "Down"]
    glyburide: Literal["No", "Steady", "Up", "Down"]
    glipizide: Literal["No", "Steady", "Up", "Down"]
    insulin: Literal["No", "Steady", "Up", "Down"]
    glyburide_metformin: Literal["No", "Steady", "Up", "Down"]
    glipizide_metformin: Literal["No", "Steady", "Up", "Down"]
    glimepiride_pioglitazone: Literal["No", "Steady", "Up", "Down"]
    metformin_rosiglitazone: Literal["No", "Steady", "Up", "Down"]
    metformin_pioglitazone: Literal["No", "Steady", "Up", "Down"]
    
API_URL = "http://localhost:8000/predict"

st.title("📊 Hospital Readmission Predictor")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    # First column
    with col1:
        patient_nbr = st.text_input("Patient Number (patient_nbr)")
        encounter_id = st.text_input("Encounter ID (encounter_id)")
        age = st.selectbox("Age", PatientInput.__annotations__['age'].__args__)
        admission_type_id = st.number_input("Admission Type ID", min_value=1, max_value=9)
        discharge_disposition_id = st.number_input("Discharge Disposition ID", min_value=1)
        admission_source_id = st.number_input("Admission Source ID", min_value=1)
        time_in_hospital = st.slider("Time in Hospital (days)", 1, 20)
        num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0)
        num_procedures = st.number_input("Number of Procedures", min_value=0)
        num_medications = st.number_input("Number of Medications", min_value=0)

    # Second column
    with col2:
        number_outpatient = st.number_input("Number of Outpatient Visits", min_value=0)
        number_emergency = st.number_input("Number of Emergency Visits", min_value=0)
        number_inpatient = st.number_input("Number of Inpatient Visits", min_value=0)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16)

        max_glu_serum = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])
        A1Cresult = st.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"])
        change = st.selectbox("Change in Medication", ["Ch", "No"])
        diabetesMed = st.selectbox("On Diabetes Medication?", ["Yes", "No"])
        diag_1 = st.text_input("Diagnosis Code 1")
        diag_2 = st.text_input("Diagnosis Code 2")
        diag_3 = st.text_input("Diagnosis Code 3")

    # Medication selection
    st.subheader("Medications")
    def med_selectbox(name):
        return st.selectbox(name, ["No", "Steady", "Up", "Down"])

    metformin = med_selectbox("Metformin")
    repaglinide = med_selectbox("Repaglinide")
    glimepiride = med_selectbox("Glimepiride")
    pioglitazone = med_selectbox("Pioglitazone")
    rosiglitazone = med_selectbox("Rosiglitazone")
    glyburide = med_selectbox("Glyburide")
    glipizide = med_selectbox("Glipizide")
    insulin = med_selectbox("Insulin")
    glyburide_metformin = med_selectbox("Glyburide-Metformin")
    glipizide_metformin = med_selectbox("Glipizide-Metformin")
    glimepiride_pioglitazone = med_selectbox("Glimepiride-Pioglitazone")
    metformin_rosiglitazone = med_selectbox("Metformin-Rosiglitazone")
    metformin_pioglitazone = med_selectbox("Metformin-Pioglitazone")

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Assemble data into a dictionary
        patient_data = {
            "patient_nbr": patient_nbr,
            "encounter_id": encounter_id,
            "age": age,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "number_diagnoses": number_diagnoses,
            "max_glu_serum": max_glu_serum,
            "A1Cresult": A1Cresult,
            "gender": gender,
            "race": race,
            "change": change,
            "diabetesMed": diabetesMed,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
            "metformin": metformin,
            "repaglinide": repaglinide,
            "glimepiride": glimepiride,
            "pioglitazone": pioglitazone,
            "rosiglitazone": rosiglitazone,
            "glyburide": glyburide,
            "glipizide": glipizide,
            "insulin": insulin,
            "glyburide_metformin": glyburide_metformin,
            "glipizide_metformin": glipizide_metformin,
            "glimepiride_pioglitazone": glimepiride_pioglitazone,
            "metformin_rosiglitazone": metformin_rosiglitazone,
            "metformin_pioglitazone": metformin_pioglitazone,
        }

        try:
            # Convert to Pydantic model
            patient_input = PatientInput(**patient_data)

            # Make prediction request
            response = requests.post(API_URL, json=patient_input.model_dump())

            if response.status_code == 200:
                prediction = response.json()
                st.write(f"Predicted Readmission: {prediction['prediction']}")
            else:
                st.error(f"Error: {response.status_code}")
        except ValidationError as e:
            st.error(f"Validation error: {e}")
