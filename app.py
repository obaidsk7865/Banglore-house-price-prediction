import streamlit as st
import pickle
import pandas as pd
import sklearn
import numpy as np

#import the model

pipe = pickle.load(open('RidgeModel.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))


st.title("House Predicton")

#brand
location = st.selectbox("Location",sort(df['location'].unique()))

# type of laptop
bhk = st.selectbox("BHK",df['bhk'].unique())

#Ram
bathrooms = st.selectbox("Bath",[1,2,3,4])

# Weight
squarefeet = st.number_input('Enter Square feet')

if st.button('Predict Price'):
    #query
    input = pd.DataFrame([[location,squarefeet,bathrooms,bhk]],columns=['location','total_sqft','bath','bhk'])

    # query = query.reshape(1,12)
    st.title("The predicted price is " + str(int(pipe.predict(input)[0])) +'L')
