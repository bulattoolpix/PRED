


import streamlit as st
import pandas as pd
import numpy as np


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