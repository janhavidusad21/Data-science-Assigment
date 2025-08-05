# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 18:53:46 2025

@author: Janhavi
"""


import streamlit as st
import pickle
import pandas as pd

# Load the saved pipeline
with open(r"C:\Users\Janhavi\OneDrive\Desktop\project deployment\log_reg (2).pkl", 'rb') as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Prediction App")

# User inputs
Pclass = st.selectbox('Passenger Class', [1, 2, 3])
Sex = st.selectbox('Sex', ['male', 'female'])
Embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])
Age = st.number_input('Age', min_value=0.0, max_value=100.0, value=25.0)
SibSp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
Parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
Fare = st.number_input('Passenger Fare', min_value=0.0, value=50.0)

# Prepare input DataFrame
input_df = pd.DataFrame({
    'Pclass': [Pclass],
    'Sex': [Sex],
    'Embarked': [Embarked],
    'Age': [Age],
    'SibSp': [SibSp],
    'Parch': [Parch],
    'Fare': [Fare]
})

# Predict
if st.button('Predict Survival'):
    prediction = model.predict(input_df)
    result = '🎉 Survived!' if prediction[0] == 1 else '😢 Did Not Survive'
    st.success(f"Prediction: {result}")
