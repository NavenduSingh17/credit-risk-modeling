#---------------------------------------------------------------------------------------------------------
# Author : NAVENDU SINGH
#---------------------------------------------------------------------------------------------------------

import streamlit as st
from prediction_helper import predict

st.markdown("### CREDIT RISK MODELING")
st.write("")

categorical_options = {
    'Residence Type': ['Mortgage', 'Owned', 'Rented'],
    'Loan Purpose': ['Auto', 'Education', 'Home', 'Personal'],
    'Loan Type': ['Secured', 'Unsecured']
}

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)
st.write("")
st.write("")

with row1[0]:
    age = st.number_input('Age', min_value=18, step=1, max_value=100, value=40)
with row1[1]:
    number_of_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2)
with row1[2]:
    loan_tenure_months = st.number_input('Loan Tenure (Months)', min_value=0, step=1, value=96)
    
with row2[0]:
    income = st.number_input('Annual Income', min_value=0, value=1500000)
with row2[1]:
    loan_amount = st.number_input('Loan Amount', min_value=0, value=2000000)
loan_to_income = loan_amount/income if income>0 else 0
with row2[2]:
    st.text("Loan to Income Ratio:")
    st.text(f"{loan_to_income:.2f}")  # Display as a text field

with row3[0]:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio (%)', min_value=0, max_value=100, step=1, value=40)
with row3[1]:
    delinquency_ratio = st.number_input('Delinquency Ratio (%)', min_value=0, max_value=100, step=1, value=12)
with row3[2]:
    avg_dpd_per_delinquency = st.number_input('Avg DPD', min_value=0, value=16)
    
with row4[0]:
    residence_type = st.selectbox('Residence Type', categorical_options['Residence Type'], index=2)
with row4[1]:
    loan_purpose = st.selectbox('Loan Purpose', categorical_options['Loan Purpose'], index=3)
with row4[2]:
    loan_type = st.selectbox('Loan Type', categorical_options['Loan Type'], index=1)

# Create a dictionary for input values
input_dict = {
    'Age': age,
    'Open Loan Accounts': number_of_open_accounts,
    'Loan Tenure': loan_tenure_months,
    'Loan to Income Ratio': loan_to_income,
    'Credit Utilization Ratio': credit_utilization_ratio,
    'Delinquency Ratio': delinquency_ratio,
    'Avg DPD': avg_dpd_per_delinquency,
    'Residence Type': residence_type,
    'Loan Purpose': loan_purpose,
    'Loan Type': loan_type
}

# Button to calculate risk
if st.button('Calculate Risk'):
    default_probability, credit_score, rating = predict(input_dict)

    st.success(f"Default Risk : {default_probability*100:.2f} %")
    st.success(f"Internal Credit Score : {credit_score}")
    st.success(f"Credit Rating : {rating}")
