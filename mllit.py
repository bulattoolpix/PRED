import streamlit as st 
import streamlit.components.v1 as stc 
import pandas as pd
import numpy as np
from PIL import Image
import requests
import io
from io import StringIO


from functionforDownloadButtons import download_button
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split


  
# importing the random forest classifier model and training it on the dataset

##st.cache(allow_output_mutation=True)
@st.cache

def XGB_train_metrics(df, params_set):
    scaler = MinMaxScaler()  
    dfx = df.iloc[:, :-1]   ##gоследняя колонка классы  (отбрасывается
    X = scaler.fit_transform(dfx)
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
  
    
    model_xgb = XGBClassifier(max_depth=params_set[0], eta=params_set[1], min_child_weight=params_set[2],
                              subsample=params_set[3], colsample_bylevel=params_set[4], colsample_bytree=params_set[5])
    # model = XGBClassifier()
    model_xgb.fit(X_train, y_train)

    # Make predictions for test data
    y_pred = model_xgb.predict(X_test)

    # Evaluate predictions
    accuracy_xgb = accuracy_score(y_test, y_pred)
    f1_xgb = f1_score(y_test, y_pred)
    roc_auc_xgb = roc_auc_score(y_test, y_pred)
    recall_xgb = recall_score(y_test, y_pred)
    precision_xgb = precision_score(y_test, y_pred)
    return accuracy_xgb, f1_xgb, roc_auc_xgb, recall_xgb, precision_xgb, model_xgb








  
  
