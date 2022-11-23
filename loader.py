


import streamlit as st
import pandas as pd
import numpy as np

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


st.write(
    """
# 📊 A/B Testing App
Upload your experiment results to see the significance of your A/B test.
"""
)

uploaded_file = st.file_uploader("Upload CSV", type=".csv")


ab_default = None
result_default = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("### Data preview")
    st.dataframe(df.head())
    csv = convert_df(df)

 st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='large_df.csv',
    mime='text/csv')
