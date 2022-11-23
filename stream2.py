import streamlit as st 
import streamlit.components.v1 as stc 
import pandas as pd
import numpy as np
from PIL import Image
import requests
import io
from io import StringIO

from functionforDownloadButtons import download_button


st.write(
    """
# 📊 A/B Testing App
Upload your experiment results to see the significance of your A/B test.
"""
)

uploaded_file = st.file_uploader(
        "",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )

    if uploaded_file is not None:
        file_container = st.expander("Check your uploaded .csv")
        df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        file_container.write(df)

    else:
        st.info(
            f"""
                👆 Upload a .csv file first. Sample to try: [biostats.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv)
                """
        )

        st.stop()
  


   
df.drop('Id', axis = 1, inplace = True)
# Renaming the target column into numbers to aid training of the model
df['Species']= df['Species'].map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})
  
# splitting the data into the columns which need to be trained(X) and the target column(y)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
  
# splitting data into training and testing data with 30 % of data as testing data respectively
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
  
# importing the random forest classifier model and training it on the dataset
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)



  
    
    
# defining the function which will make the prediction using 
# the data which the user inputs
def prediction(sepal_length, sepal_width, petal_length, petal_width):  
   
    prediction = classifier.predict(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    print(prediction)
    return prediction
      
# this is the main function in which we define our webpage 

def main():
    
      # giving the webpage a title
        
    st.title("Iris Flower Prediction")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:WHITE;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    sepal_length = st.text_input("Sepal Length", "Type Here")
    sepal_width = st.text_input("Sepal Width", "Type Here")
    petal_length = st.text_input("Petal Length", "Type Here")
    petal_width = st.text_input("Petal Width", "Type Here")
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(sepal_length, sepal_width, petal_length, petal_width)
    st.success('The output is {}'.format(result))


     
if __name__=='__main__':
    main()
    
    


    CSVButton = download_button(
        df,
        "File.csv",
        "Download to CSV",
    )

    


