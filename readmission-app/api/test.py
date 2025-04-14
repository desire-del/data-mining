import joblib
import pandas as pd
from utils import prepare_patient_vector

full = pd.read_csv("data/historic_data.csv")

test_patient = {
    'encounter_id': 123456,
    'patient_nbr': 78910,
    'race': 'Caucasian',
    'gender': 'M',
    'age': 45,
    'admission_type_id': 1,
    'discharge_disposition_id': 2,
    'admission_source_id': 3,
    'time_in_hospital': 4,
    'medical_specialty': 'Cardiology',
    'num_lab_procedures': 10,
    'num_procedures': 2,
    'num_medications': 5,
    'number_outpatient': 1,
    'number_emergency': 0,
    'number_inpatient': 1,
    'diag_1': 'I10',
    'diag_2': 'E11',
    'diag_3': 'M54',
    'number_diagnoses': 3,
    'max_glu_serum': 'None',
    'A1Cresult': 'None',
    'metformin': 1,
    'repaglinide': 0,
    'nateglinide': 0,
    'chlorpropamide': 0,
    'glimepiride': 0,
    'acetohexamide': 0,
    'glipizide': 1,
    'glyburide': 0,
    'tolbutamide': 0,
    'pioglitazone': 0,
    'rosiglitazone': 0,
    'acarbose': 0,
    'miglitol': 0,
    'troglitazone': 0,
    'tolazamide': 0,
    'examide': 0,
    'citoglipton': 0,
    'insulin': 1,
    'glyburide-metformin': 0,
    'glipizide-metformin': 0,
    'glimepiride-pioglitazone': 0,
    'metformin-rosiglitazone': 0,
    'metformin-pioglitazone': 0,
    'change': 0,
    'diabetesMed': 1,
    'readmitted': 0,
    'admission_type': 'Elective',
    'discharge_disposition': 'Home',
    'admission_source': 'Physician'
}
t = prepare_patient_vector(test_patient, full)

model = joblib.load("models/pipeline_xgb.pkl")
t = pd.DataFrame(data = [list(t)], columns=list(t.index))
print(t.isna().sum())

print(model.predict_proba(t))