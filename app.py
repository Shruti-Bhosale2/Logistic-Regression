import streamlit as st
import pickle
import numpy as np

st.title("Diabetes Prediction using Logistic Regression")

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.header("Enter Patient Details")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("The person is likely to have Diabetes")
    else:
        st.success("The person is not likely to have Diabetes")

