def drop_generated_column(input_df):
    return input_df.drop(columns=['loan_purpose_Home'])