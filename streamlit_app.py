# income_predictor_ui.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and label encoders
@st.cache_resource
def load_model():
    model = joblib.load("model_lightgbm_v2.pkl")
    return model

model = load_model()

st.set_page_config(page_title="Income Predictor", layout="centered")
st.title("💰 Income Prediction App")
st.markdown("Fill in the details below to predict if income is more than 50K.")

# --- User Input Form ---
with st.form("user_form"):
    st.subheader("📋 Personal Information")

    age = st.slider("Age", 18, 90, 30)
    workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                                           "Local-gov", "State-gov", "Without-pay", "Never-worked"])
    education = st.selectbox("Education", ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", 
                                           "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", 
                                           "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
    marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married", 
                                                     "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
    occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", 
                                             "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
                                             "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
                                             "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
    relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", 
                                                 "Other-relative", "Unmarried"])
    race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
    sex = st.radio("Sex", ["Male", "Female"])
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0, step=100)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0, step=100)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)
    native_country = st.selectbox("Native Country", ["United-States", "Mexico", "Philippines", "Germany", 
                                                     "Canada", "India", "China", "Cuba", "Iran", "England", 
                                                     "Other"])

    submitted = st.form_submit_button("Predict Income")

# --- Manual Encoding based on training ---
def manual_encode(data_dict):
    # Label Encoding map (same as training time)
    mappings = {
        "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                      "Local-gov", "State-gov", "Without-pay", "Never-worked"],
        "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", 
                      "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", 
                      "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
        "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", 
                           "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
        "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", 
                       "Exec-managerial", "Prof-specialty", "Handlers-cleaners", 
                       "Machine-op-inspct", "Adm-clerical", "Farming-fishing", 
                       "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
        "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", 
                         "Other-relative", "Unmarried"],
        "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
        "sex": ["Male", "Female"],
        "native-country": ["United-States", "Mexico", "Philippines", "Germany", 
                           "Canada", "India", "China", "Cuba", "Iran", "England", "Other"]
    }

    encoded = []
    for feature, categories in mappings.items():
        value = data_dict[feature]
        encoded.append(categories.index(value) if value in categories else 0)

    # Add numerical features
    encoded_data = [
        data_dict["age"],
        *encoded,
        data_dict["capital-gain"],
        data_dict["capital-loss"],
        data_dict["hours-per-week"]
    ]
    return np.array(encoded_data).reshape(1, -1)

# --- Prediction ---
if submitted:
    user_input = {
        "age": age,
        "workclass": workclass,
        "education": education,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }

    encoded_input = manual_encode(user_input)
    prediction = model.predict(encoded_input)[0]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.success("Predicted Income: **>50K**")
    else:
        st.info("Predicted Income: **<=50K**")
