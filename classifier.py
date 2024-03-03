import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""# Simple Iris Flower Prediction App.  
         This app predicts whether the flower is Iris or not !!!""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('sepal_length',4.3,7.9,5.4)
    sepal_width = st.sidebar.slider('sepal_width',2.2,4.4,5.4)
    petal_length = st.sidebar.slider('petal_length',1.4,6.9,1.4)
    petal_width = st.sidebar.slider('petal_width',0.1,2.5,0.5)
    data =  {'sepal_length':sepal_length
             ,'sepal_width':sepal_width
             ,'petal_length':petal_length
             ,'petal_width':petal_width}
    
    features = pd.DataFrame(data,index = [0])
    return features

df = user_input_features()

st.subheader('User Input')
st.write(df)

iris = datasets.load_iris()

X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X,y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class Labels and their corresponding index')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)