import pandas as pd

def engineer_top_features(df):
    df = df.copy()

    # 1. Indicateur d'expiration (décès)
    df['expiration_ind'] = df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21]).astype(int)

    # 2. Médicaments importants (compte et intensité)
    important_drugs = ['metformin', 'repaglinide', 'glimepiride', 'pioglitazone', 
                       'rosiglitazone', 'glyburide', 'glipizide', 'insulin']
    
    df['active_med_count'] = df[important_drugs].apply(lambda x: (x != 'No').sum(), axis=1)
    df['med_intensity'] = df[important_drugs].apply(lambda x: sum(1 for m in x if m in ['Up', 'Down']), axis=1)

    # 3. Features composites
    df['urgent_care_ratio'] = (df['number_emergency'] + 1) / (df['number_outpatient'] + 1)
    df['stay_procedure_ratio'] = df['time_in_hospital'] / (df['num_procedures'] + 1)
    df['med_complexity_score'] = (df['num_medications'] * 0.5 + df['active_med_count'] * 0.3 + df['med_intensity'] * 0.2)

    # 4. Diagnostiques (classification des codes)
    def map_diag_code(diag):
        try:
            code = float(diag)
        except:
            return 'other'
        if 250 <= code < 251: return 'diabetes'
        elif 390 <= code <= 459: return 'cardiovascular'
        elif 580 <= code <= 629: return 'renal'
        elif code in [3051, 303, 305]: return 'substance_abuse'
        else: return 'other'
    
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[f'{col}_group'] = df[col].apply(map_diag_code)

    # 5. Abnormal stay (séjour anormal)
    df['abnormal_stay'] = pd.cut(df['time_in_hospital'], 
                                 bins=[0, 2, 5, 10, float('inf')], 
                                 labels=['very_short', 'short', 'normal', 'long'])

    # 6. Instabilité thérapeutique : changement de traitement
    df['therapy_instability'] = (df[important_drugs].isin(['Up', 'Down']).sum(axis=1) > 1).astype(int)

    # 7. Comorbidité et variabilité du diagnostic
    df['comorbidity_count'] = df[['diag_1_group', 'diag_2_group', 'diag_3_group']].nunique(axis=1)
    df['diag_variability'] = (df['diag_1'] != df['diag_2']).astype(int) + (df['diag_2'] != df['diag_3']).astype(int)

    df['readmitted_lt30_ind'] = ( df['readmitted']=='<30' ).astype(int)
    df['readmitted_gt30_ind'] = ( df['readmitted']=='>30' ).astype(int)
    df['readmitted_no_ind'] = ( df['readmitted']=='NO' ).astype(int)
    df['readmitted_ind'] = df['readmitted_lt30_ind'] + df['readmitted_gt30_ind']
    readmission_history_features = [
        'mb_readmitted_gt30_ct', 'mb_readmitted_no_ct',
        'prior_30day_readmits', 'mb_readmitted_lt30_ct'
    ]
    
    # Si l’une d’elles est absente dans df, on la met à 0 (utile pour nouveaux patients)
    for col in readmission_history_features:
        if col not in df.columns:
            df[col] = 0

    # 9. Sélection des features finales
    final_features = [
        # Variables clés pour le modèle
        'expiration_ind', 'prior_30day_readmits', 'total_prior_admissions',
        'number_inpatient', 'number_emergency', 'number_outpatient',
        'number_diagnoses', 'time_in_hospital', 'num_procedures', 'num_medications',
        'num_lab_procedures', 'urgent_care_ratio', 'stay_procedure_ratio',
        'med_complexity_score', 'active_med_count', 'med_intensity',
        'therapy_instability', 'comorbidity_count', 'diag_variability',
        'avg_stay_duration', 'max_stay_duration', 'lifetime_inpatient_visits',
        'lifetime_emergency_visits', 'encounter_ct', 'mb_time_in_hospital',
        'mb_readmitted_lt30_ct', 'mb_readmitted_gt30_ct', 'mb_readmitted_no_ct',
        'mb_num_lab_procedures_ct', 'mb_num_procedures_ct', 'mb_num_medications_ct',
        'mb_number_outpatient_ct', 'mb_number_emergency_ct', 'mb_number_inpatient_ct',
        'mb_number_diagnoses_ct', 'major_chronic_diag',

        # Variables catégorielles
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'age', 'race', 'gender', 'max_glu_serum', 'A1Cresult',
        'metformin', 'repaglinide', 'glimepiride', 'pioglitazone',
        'rosiglitazone', 'glyburide', 'glipizide', 'insulin',
        'diag_1_group', 'diag_2_group', 'diag_3_group', 'abnormal_stay',
        'change', 'diabetesMed'
    ]

    # Ajout des features patient-level (mb_)
    df['encounter_ct'] = df['patient_nbr'].map(df.groupby('patient_nbr')['encounter_id'].nunique())
    df['mb_time_in_hospital'] = df.groupby('patient_nbr')['time_in_hospital'].transform('sum')
    df['mb_readmitted_lt30_ct'] = df.groupby('patient_nbr')['readmitted_lt30_ind'].transform('sum')
    df['mb_readmitted_gt30_ct'] = df.groupby('patient_nbr')['readmitted_gt30_ind'].transform('sum')
    df['mb_readmitted_no_ct'] = df.groupby('patient_nbr')['readmitted_no_ind'].transform('sum')
    df['mb_num_lab_procedures_ct'] = df.groupby('patient_nbr')['num_lab_procedures'].transform('sum')
    df['mb_num_procedures_ct'] = df.groupby('patient_nbr')['num_procedures'].transform('sum')
    df['mb_num_medications_ct'] = df.groupby('patient_nbr')['num_medications'].transform('sum')
    df['mb_number_outpatient_ct'] = df.groupby('patient_nbr')['number_outpatient'].transform('sum')
    df['mb_number_emergency_ct'] = df.groupby('patient_nbr')['number_emergency'].transform('sum')
    df['mb_number_inpatient_ct'] = df.groupby('patient_nbr')['number_inpatient'].transform('sum')
    df['mb_number_diagnoses_ct'] = df.groupby('patient_nbr')['number_diagnoses'].transform('sum')

    # Droper les features inutiles
    to_drop = ['readmitted_lt30_ind', 'readmitted_gt30_ind', 'readmitted_no_ind']
    df = df.drop(columns=to_drop, errors='ignore')

    # Retourner uniquement les features sélectionnées
    df = df[[col for col in final_features if col in df.columns]]
    
    return df


# === Préparation du vecteur patient pour la prédiction ===
def prepare_patient_vector(new_patient_dict, full_df):
    """
    Combine le nouveau patient avec le dataset complet et applique le feature engineering.

    Args:
        new_patient_dict (dict): Nouveau patient (features de base)
        full_df (pd.DataFrame): Historique patient avec les colonnes de base

    Returns:
        Series avec les features transformées prêtes à l’inférence
    """
    new_patient_df = pd.DataFrame([new_patient_dict])
    combined_df = pd.concat([full_df, new_patient_df], ignore_index=True)
    transformed_df = engineer_top_features(combined_df)
    return transformed_df.iloc[-1]
