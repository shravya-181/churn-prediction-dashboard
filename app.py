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

# =========================== OVERVIEW ===========================
if page == "Overview":
    st.header("Overview")
    st.info("How to read this: these numbers summarise all learners on the platform.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total learners", f"{len(df):,}")
    c2.metric("Churn rate", f"{df['churned'].mean() * 100:.1f}%")
    c3.metric("Avg satisfaction", f"{df['satisfaction_score'].mean():.2f} / 5")
    c4.metric("Avg tenure", f"{df['tenure_months'].mean():.1f} months")

    st.subheader("Churn rate by plan type (%)")
    plan_churn = df.groupby('plan_type')['churned'].mean() * 100
    st.bar_chart(plan_churn)
    st.success(
        f"Takeaway: {plan_churn.idxmax()} learners churn the most "
        f"({plan_churn.max():.1f}%), compared with {plan_churn.min():.1f}% for {plan_churn.idxmin()}."
    )

    st.subheader("Churn rate by engagement trend (%)")
    trend_labels = {-1: 'Declining', 0: 'Flat', 1: 'Improving'}
    trend_churn = df.groupby('engagement_trend')['churned'].mean() * 100
    trend_churn.index = trend_churn.index.map(trend_labels)
    st.bar_chart(trend_churn)
    st.success(
        f"Takeaway: learners whose activity is declining churn at {trend_churn['Declining']:.1f}%, "
        f"versus {trend_churn['Improving']:.1f}% for learners whose activity is improving."
    )

# ============================ EXPLORE ============================
elif page == "Explore":
    st.header("Explore: what makes learners leave?")
    st.info("How to read this: pick a factor, and compare how many learners leave in each group.")

    st.subheader("1. Churn by learner group")
    groups = {
        "Plan type": "plan_type",
        "Occupation": "occupation_type",
        "Signup goal": "signup_goal",
        "Used a signup discount": "signup_discount_used",
    }
    group_label = st.selectbox("Compare churn by:", list(groups.keys()))
    rates = df.groupby(groups[group_label])['churned'].mean() * 100
    st.bar_chart(rates)
    st.success(
        f"Takeaway: '{rates.idxmax()}' learners churn the most ({rates.max():.1f}%), "
        f"while '{rates.idxmin()}' learners churn the least ({rates.min():.1f}%)."
    )

    st.subheader("2. Behaviour of learners who stayed vs left")
    behaviours = {
        "Course completion (%)": "course_completion_rate",
        "Average quiz score": "avg_quiz_score",
        "Days since last login": "days_since_last_login",
        "Hands-on exercises completed": "hands_on_exercises_completed",
        "Satisfaction (1-5)": "satisfaction_score",
        "Forum posts": "forum_posts",
    }
    behaviour_label = st.selectbox("Pick a behaviour:", list(behaviours.keys()))
    avg = df.groupby('churned')[behaviours[behaviour_label]].mean()
    compare = pd.DataFrame({"Average": [avg[0], avg[1]]}, index=["Stayed", "Churned"])
    st.bar_chart(compare)
    st.success(
        f"Takeaway: on '{behaviour_label}', learners who stayed average {avg[0]:.1f}, "
        f"while learners who churned average {avg[1]:.1f}."
    )

# ============================ PREDICT ============================
elif page == "Predict":
    st.header("Predict")
    st.write("This page will let you input a learner and get a risk score.")

# ============================== WHY? ==============================
elif page == "Why?":
    st.header("Why?")
    st.write("This page will show SHAP explanations.")

# ======================= RETENTION SIMULATOR =======================
elif page == "Retention Simulator":
    st.header("Retention Simulator")
    st.write("This page will show the ROI simulator.")
