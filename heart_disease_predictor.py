import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('best_model.pkl')

# Define the app
st.title('Heart Disease Prediction')
st.write('Enter the patient details to predict the likelihood of heart disease.')

# Input fields for patient data
age = st.number_input('Age', min_value=1, max_value=120, value=30)
sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
cp = st.number_input('Chest Pain Type', min_value=0, max_value=3, value=0)
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=80, max_value=200, value=120)
chol = st.number_input('Serum Cholesterol (chol)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
restecg = st.number_input('Resting Electrocardiographic Results (restecg)', min_value=0, max_value=2, value=1)
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.number_input('Slope of the Peak Exercise ST Segment (slope)', min_value=0, max_value=2, value=1)
ca = st.number_input('Number of Major Vessels Colored by Flouroscopy (ca)', min_value=0, max_value=4, value=0)
thal = st.number_input('Thalassemia (thal)', min_value=0, max_value=3, value=2)

# Convert inputs into a dataframe
input_data = pd.DataFrame({
    'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps], 'chol': [chol], 'fbs': [fbs],
    'restecg': [restecg], 'thalach': [thalach], 'exang': [exang], 'oldpeak': [oldpeak],
    'slope': [slope], 'ca': [ca], 'thal': [thal]
})

# Predict and display result
if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]  # Probability of positive class

    if prediction == 1:
        st.error(f'The patient is likely to have heart disease. Probability: {prediction_proba:.2f}')
    else:
        st.success(f'The patient is not likely to have heart disease. Probability: {prediction_proba:.2f}')

# Add some information about the app
st.write("""
### About the App
This application uses a machine learning model to predict the likelihood of heart disease based on patient data. Please fill in all the fields with the patient's information to get a prediction.
""")
