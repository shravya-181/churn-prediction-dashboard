import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="EdTech Churn Dashboard", layout="wide")

# ---- Load everything once, at startup ----
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('churn_scaler.pkl')
    kmeans = joblib.load('persona_kmeans.pkl')
    cluster_scaler = joblib.load('persona_scaler.pkl')
    with open('model_columns.json') as f:
        model_columns = json.load(f)
    with open('cluster_columns.json') as f:
        cluster_columns = json.load(f)
    with open('persona_info.json') as f:
        persona_info = json.load(f)
    df = pd.read_csv('edtech_churn_dataset.csv')
    return model, scaler, kmeans, cluster_scaler, model_columns, cluster_columns, persona_info, df

model, scaler, kmeans, cluster_scaler, model_columns, cluster_columns, persona_info, df = load_artifacts()

# ---- Sidebar navigation ----
page = st.sidebar.radio("Navigate", ["Overview", "Explore", "Predict", "Why?", "Retention Simulator"])

st.title("📚 EdTech Learner Churn Dashboard")

if page == "Overview":
    st.header("Overview")
    st.write("This page will show key stats.")

elif page == "Explore":
    st.header("Explore")
    st.write("This page will show charts.")

elif page == "Predict":
    st.header("Predict")
    st.write("This page will let you input a learner and get a risk score.")

elif page == "Why?":
    st.header("Why?")
    st.write("This page will show SHAP explanations.")

elif page == "Retention Simulator":
    st.header("Retention Simulator")
    st.write("This page will show the ROI simulator.")
