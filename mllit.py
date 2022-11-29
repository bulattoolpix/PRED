import streamlit as st 
import streamlit.components.v1 as stc 
import pandas as pd
import numpy as np
from PIL import Image
import requests
import io
from io import StringIO
import datetime
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import base64
##from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA

from functionforDownloadButtons import download_button
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


  
# importing the random forest classifier model and training it on the dataset

##st.cache(allow_output_mutation=True)

    
@st.cache

 
def upload_different_data(uploaded_file):
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    df = pd.read_csv(uploaded_file, low_memory=False)
    df.iloc[:, -1]
    rows = df.shape[0]
    columns = df.shape[1]
    
    # Drop rows with all Null
    df = df.fillna(0)
    data=df ##,  = data_preprocessing(df)
    return df, data,  'Uploaded file', rows, columns

 
def upload_different_data2(uploaded_file2):
    uploaded_file2 = st.file_uploader("Choose a CSV file", type="csv")
    df2 = pd.read_csv(uploaded_file2, low_memory=False)
    df2.iloc[:, -1]
    rows2 = df2.shape[0]
    columns2 = df2.shape[1]
    
    # Drop rows with all Null
    df2 = df2.fillna(0)
    data2=df2 ##,  = data_preprocessing(df)
    return df2, data2,  'Uploaded file', rows2, columns2
    


def XGB_train_metrics(df,params_set):
    scaler = MinMaxScaler()  
    dfx = df.iloc[:, :-1]   ##gоследняя колонка классы  (отбрасывается
    X = scaler.fit_transform(dfx)
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
 
    
    model_xg = XGBClassifier(max_depth=params_set[0], eta=params_set[1], min_child_weight=params_set[2],
                              subsample=params_set[3], colsample_bylevel=params_set[4], colsample_bytree=params_set[5])
    # model = XGBClassifier()
    model_xg.fit(X_train, y_train)

    # Make predictions for test data
    y_pred = model_xg.predict(X_test)

    # Evaluate predictions
    accuracy_xgb = accuracy_score(y_test, y_pred)
    f1_xgb = f1_score(y_test, y_pred,average='micro')
    ##roc_auc_xgb = roc_auc_score(y_test, y_pred,multi_class='ovr')
    recall_xgb = recall_score(y_test, y_pred,average='micro')
    precision_xgb = precision_score(y_test, y_pred,average='micro')
    return accuracy_xgb, f1_xgb,recall_xgb, precision_xgb, model_xg ##roc_auc_xgb, 
 









 

  
def feature_summary(data):
    print('DataFrame shape')
    print('rows:', data.shape[0])
    print('cols:', data.shape[1])
    col_list = ['Null', 'Unique_Count', 'Data_type',
                'Max/Min', 'Mean', 'Std', 'Skewness', 'Sample_values']
    df = pd.DataFrame(index=data.columns, columns=col_list)
    df['Null'] = list([len(data[col][data[col].isnull()])
                       for i, col in enumerate(data.columns)])
    df['Unique_Count'] = list([len(data[col].unique())
                               for i, col in enumerate(data.columns)])
    df['Data_type'] = list(
        [data[col].dtype for i, col in enumerate(data.columns)])
    for i, col in enumerate(data.columns):
        if 'float' in str(data[col].dtype) or 'int' in str(data[col].dtype):
            df.at[col, 'Max/Min'] = str(round(data[col].max(), 2)) + '/' + str(round(data[col].min(), 2))
            df.at[col, 'Mean'] = data[col].mean()
            df.at[col, 'Std'] = data[col].std()
            df.at[col, 'Skewness'] = data[col].skew()
        df.at[col, 'Sample_values'] = list(data[col].unique())

    return(df.fillna('-'))
  
##скачивание пердикшн 
def prediction_downloader(data2):
    st.write('')
    st.subheader('Want to download the prediction results?')
    csv = data2.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    st.markdown(href, unsafe_allow_html=True)
    
  
  
  
  
  
  
  
  
def home_page_builder( df, data, rows, columns):
 
    st.title("Streamlit Demo")

    st.subheader('INTRODUCTION')
    st.write('')
    st.write(
        'Using machine learning algorithms to predict approval status of application')
    st.write('')
    st.write('')

   
    # Insert Check-Box to show the snippet of the data.
    if  st.checkbox('Show Data'):
        st.subheader("Raw data")
        st.write(
            f'Input dataset includes **{rows}** rows and **{columns}** columns')
      ##  st.write(df.head())
    
        st.write(data.head())

    # show data visulization
    if st.checkbox('Show Visualization'):
        fig = px.histogram(data.iloc[:, -1], x='Species',
                           title='Distribution of Target Variable "')
        st.plotly_chart(fig)
        st.write('We can see Approved is about three times of Decliened, which may bring an imbalanced issue for prediction - we will deal with this issue during modeling.')
        st.write('-'*60)
      
      ##второй рисунок подряд 
      ## fig = px.histogram(df.time, x='time',
       ##                    title='Distribution of Date Time')
       ## st.plotly_chart(fig)
       ## st.write(
       ##     'The distribution of date time, we can see most of the data are from recent two months.')
       ## st.write('-'*60)

    # Show feature summary
    if st.checkbox('Show Feature Summary'):
        st.write('Raw data after dropping rows that have NULL for every column; ')
        st.write('Also converted column "time" to datetime format')
        st.write(feature_summary(data))
        st.write('For each columns in our original dataset, we can see the statistics summary (Null Value Count, Unique Value Count, Data Type, etc.)')


  
  
  

def xgb_predictor(df,data2,params_set ):
    scaler = MinMaxScaler()  
    dfx1 = df.iloc[:, :-1]   ##gоследняя колонка классы  (отбрасывается
    X1= scaler.fit_transform(dfx1)
    Xzero= scaler.fit_transform(data2)
    y1 = df.iloc[:, -1]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25, random_state = 0)
 
    
    model_xgb2 = XGBClassifier(max_depth=params_set[0], eta=params_set[1], min_child_weight=params_set[2],
                              subsample=params_set[3], colsample_bylevel=params_set[4], colsample_bytree=params_set[5])
    # model = XGBClassifier()
    model_xgb2.fit(X_train1, y_train1)
    

    # Make predictions for test data
    data2['target'] =  model_xgb2.predict(Xzero)
    
    df_feature = pd.DataFrame.from_dict(model_xgb2.get_booster().get_fscore(), orient='index')
    df_feature.columns = ['Feature Importance']
    
    feature_importance = df_feature.sort_values(by='Feature Importance', ascending=False).T
    ##fig = px.bar(feature_importance, x=feature_importance.columns, y=feature_importance.T)
    ##fig.update_xaxes(tickangle=45, title_text='Features')
    ##fig.update_yaxes(title_text='Feature Importance')
    ##st.plotly_chart(fig)
    return data2, model_xgb2

## не работает
def featureimp (data):
    scaler = MinMaxScaler()  
    dfx1 = data.iloc[:, :-1]   ##gоследняя колонка классы  (отбрасывается
    X1= scaler.fit_transform(dfx1)
    y1 = data.iloc[:, -1]
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25, random_state = 0)
 
    model_xgb3 = XGBClassifier()
    # model = XGBClassifier()
    model_xgb3.fit(X_train1, y_train1)
    
    df_feature = pd.DataFrame.from_dict(model_xgb3.get_booster().get_fscore(), orient='index')
    df_feature.columns = ['Feature Importance']
    
    feature_importance = df_feature.sort_values(by='Feature Importance', ascending=False).T
    ##fig = px.bar(feature_importance, x=feature_importance.columns, y=feature_importance.T)
    ##fig.update_xaxes(tickangle=45, title_text='Features')
    ##fig.update_yaxes(title_text='Feature Importance')
    ##st.plotly_chart(fig)
    return  feature_importance,df_feature
       


def xgb_page_builder(data,data2 ):
    st.sidebar.header('Hyper Parameters')
    st.sidebar.markdown('You can tune the hyper parameters by siding')
    max_depth = st.sidebar.slider('Select max_depth (default = 30)', 3, 30, 30)
    eta = st.sidebar.slider(
        'Select learning rate (divided by 10) (default = 0.1)', 0.01, 1.0, 1.0)
    min_child_weight = st.sidebar.slider(
        'Select min_child_weight (default = 0.3)', 0.1, 3.0, 0.3)
    subsample = st.sidebar.slider(
        'Select subsample (default = 0.75)', 0.5, 1.0, 0.75)
    colsample_bylevel = st.sidebar.slider(
        'Select colsample_bylevel (default = 0.5)', 0.5, 1.0, 0.5)
    colsample_bytree = st.sidebar.slider(
        'Select colsample_bytree (default = 1.0)', 0.5, 1.0, 1.0)
    params_set = [max_depth, 0.1*eta, min_child_weight,
                  subsample, colsample_bylevel, colsample_bytree]

    start_time = datetime.datetime.now()
   ##roc_auc_xgb, 
    accuracy_xgb, f1_xgb,  recall_xgb, precision_xgb, model_xgb = XGB_train_metrics(data, params_set)
    
    model_xgb = XGB_train_metrics(data,params_set)
    
    model_xgb2= xgb_predictor(data,data2,params_set )   ####прогноз новой выборки на основе выставленных гипермарметров 
    ##featureimp (data)
    ##df_feature = pd.DataFrame.from_dict(model_xgb2.get_booster().get_fscore(), orient='index')
    
    st.subheader('Model Introduction')
    st.write('',params_set)
    st.write('XGBoost - e**X**treme **G**radient **B**oosting, is an implementation of gradient boosted **decision trees** designed for speed and performance, which has recently been dominating applied machine learning. We recommend you choose this model to do the prediction.')
    st.write('')
    st.subheader('XGB metrics on testing dataset')
    st.write('')
    st.write('')
    st.write('')
    st.markdown("We separated the dataset to training and testing dataset, using training data to train our model then do the prediction on testing dataset, here's XGB prediction performance: ")
    st.write('')
    st.write(
        f'Running time: {(datetime.datetime.now() - start_time).seconds} s')
    st.table(pd.DataFrame(data=[round(accuracy_xgb * 100.0, 2), round(precision_xgb * 100.0, 2), round(recall_xgb*100, 2),  round(f1_xgb*100, 2)], ##,round(roc_auc_xgb*100, 2),],
                          index=['Accuracy', 'Precision (% we predicted as Declined are truly Declined)', 'Recall (% Declined have been identified)',  'F1'], columns=['%'])) ##'ROC_AUC',
    st.subheader('Feature Importance:')
    st.write('Predicted target values for unknown target label ',data2)
    ##st.write('Predicted target values for unknown target label ', df_feature)
    # Download prediction as a CSV file
   
   
    return   model_xgb2
          # Plot feature importance
  
  
  
  

  
     
def main():
    global df
    st.markdown('<style>body{background-color: grey;}</style>',unsafe_allow_html=True)

    """Streamlit demo web app"""
    
    st.write(
    """
# 📊 AUTO CLASSIFIER App
Загрузите файл для обучения и файл для прогноза 
"""
)
    uploaded_file = st.file_uploader(
        "",
        key="1",
     
    )
    if uploaded_file is not None:
        global df
        df = pd.read_csv(uploaded_file)
        ##uploaded_file.seek(0)
    
        df, data, filename, rows, columns = upload_different_data(uploaded_file)
    
            
    uploaded_file2 = st.file_uploader(
        "",
        key="2",
     
         )      
    if uploaded_file2 is not None:
        global df2 
      
        df2 = pd.read_csv(uploaded_file2)
        ##uploaded_file2.seek(0)
        df2, data2, filename2, rows2, columns2 = upload_different_data2(uploaded_file2)
    

         
            

    st.sidebar.title('Menu')
    choose_model = st.sidebar.selectbox("Choose the page or model", [
                                        "Home",  "XGB"])    
    
    
    if choose_model == "Home":
       
       home_page_builder(  df, data, rows, columns)
       
       

    if choose_model == "XGB":
        model_xgb = xgb_page_builder(data,data2  )
        
      ##data2 = pd.read_csv(uploaded_file2)
      ##st.write('Uploaded data:', data2.head(30))
      ##scaler = MinMaxScaler() 
      ##V = scaler.fit_transform( data2 )   
        if(st.checkbox("Want to check Feature importance")):
           ##prediction_downloader(data2) ###загрузк
 ##             featureimp (df)
              scaler = MinMaxScaler()  
              dfx1 = data.iloc[:, :-1]   ##gоследняя колонка классы  (отбрасывается
              X1= scaler.fit_transform(dfx1)
              y1 = data.iloc[:, -1]
              X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.25, random_state = 0)
 
              model_xgb3 = XGBClassifier()
    # model = XGBClassifier()
              model_xgb3.fit(X_train1, y_train1)
    
              df_feature = pd.DataFrame.from_dict(model_xgb3.get_booster().get_fscore(), orient='index')
             ## df_feature.columns =dfx1.columns.values.tolist()
              df_feature.columns = ['Feature Importance']
              ##df_feature.columns =dfx1.columns
              list(dfx1.columns)
              df_feature
              

#Using list(df) to get the list of all Column Names

             ## st.bar_chart(df_feature)
    
              feature_importance=df_feature.sort_values(by='Feature Importance', ascending=False).T
              feature_importance
              dfx1.columns
              list(dfx1)
              dfx1.columns.values.tolist()
              
              sorted_idx = pd.DataFrame(model_xgb3.feature_importances_)
           
              ##sorted_idx .columns =dfx1.columns
              st.bar_chart( sorted_idx)

              ##st.bar_chart(model_xgb3.feature_importances_.rename_axis('unique_values'), label=data.columns)


        



            
        

    

            
            
            
            
if __name__ == "__main__":
    main()








  
  
