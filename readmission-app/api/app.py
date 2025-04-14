from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Literal
from utils import prepare_patient_vector


app = FastAPI(title="Hospital Readmission Predictor")

model = joblib.load("models/pipeline_xgb.pkl")
historical_df = pd.read_csv("data/historic_data.csv")



class PatientData(BaseModel):
    patient_nbr: int
    encounter_id: int
    age: str
    race: str
    gender: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    max_glu_serum: Literal["None", "Norm", ">200", ">300"]
    A1Cresult: Literal["None", "Norm", ">7", ">8"]
    metformin: str
    repaglinide: str
    glimepiride: str
    pioglitazone: str
    rosiglitazone: str
    glyburide: str
    glipizide: str
    insulin: str
    change: Literal["No", "Ch"]
    diabetesMed: Literal["Yes", "No"]
    diag_1: str
    diag_2: str
    diag_3: str


@app.post("/predict")
def predict_readmission(input_data: PatientData):

    try:
        # Convertir l'input en DataFrame
        patient_df = pd.DataFrame([input_data.model_dump()])
        
        
        # Traitement du vecteur de données patient avec la fonction prepare_patient_vector
        processed_patient_vector = prepare_patient_vector(patient_df.iloc[0], historical_df)
        
        # Prédiction avec le modèle XGBoost
        t = pd.DataFrame(data = [list(processed_patient_vector)], columns=list(processed_patient_vector.index))
       
        prediction_proba = model.predict_proba(t)[0]
        
        classes = ['<30', '>30', 'NO']
        print(np.argmax(prediction_proba))
        prediction_label = classes[np.argmax(prediction_proba)]
        print(prediction_label)

        # Retourner la prédiction et les probabilités pour chaque classe
        return {
            "prediction": prediction_label,
            "probabilities": {
                classes[0]: float(prediction_proba[0]),
                classes[1]: float(prediction_proba[1]),
                classes[2]: float(prediction_proba[2])
            }
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health_check():
    # Vérifier si l'API fonctionne correctement
    return {"status": "API is healthy"}
