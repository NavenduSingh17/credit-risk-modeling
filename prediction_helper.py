#---------------------------------------------------------------------------------------------------------
# Author : NAVENDU SINGH
#---------------------------------------------------------------------------------------------------------

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = 'artifacts/classifier.joblib'

model = joblib.load(MODEL_PATH)

def prepare_input(input_dict):
    data = {
        'age': [input_dict['Age']],     # Made it a list otherwise scalar value will be passed
        'number_of_open_accounts': [input_dict['Open Loan Accounts']],
        'loan_tenure_months': [input_dict['Loan Tenure']],
        'loan_to_income': [input_dict['Loan to Income Ratio']],
        'credit_utilization_ratio': [input_dict['Credit Utilization Ratio']],
        'delinquency_ratio': [input_dict['Delinquency Ratio']/100],
        'avg_dpd_per_delinquency': [input_dict['Avg DPD']],
        'residence_type': [input_dict['Residence Type']],
        'loan_purpose': [input_dict['Loan Purpose']],
        'loan_type': [input_dict['Loan Type']]
    }

    df = pd.DataFrame(data)
    return df
    # I could've also done df = pd.DataFrame(data, index=[0]) if I passed scalar values in the dictionary 'data'
    # By passing index=[0], Pandas knew that "Create exactly one row, and label that row index 0."


def predict(input_dict):
    input_df = prepare_input(input_dict)

    base_score = 300
    scale_length = 600

    default_probability = model.predict_proba(input_df)[0][1]
    
    non_default_probability = 1 - default_probability
    credit_score = base_score + (non_default_probability * scale_length)

    rating = get_rating(credit_score)

    return default_probability, int(credit_score), rating
    #return model.predict_proba(input_df)[0][1]

def get_rating(score):
    if 300 <= score < 500:
        return 'Poor'
    elif 500 <= score < 650:
        return 'Average'
    elif 650 <= score < 750:
        return 'Good'
    elif 750 <= score <= 900:
        return 'Excellent'
    else:
        return 'Undefined'  # in case of any unexpected score
    