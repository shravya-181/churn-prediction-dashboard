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

# Plain-English names for features (used in explanations)
FEATURE_LABELS = {
    "avg_quiz_score": "their average quiz score",
    "video_replay_count": "how often they replay videos",
    "course_completion_rate": "their course completion",
    "days_since_last_login": "the days since their last login",
    "engagement_trend": "their engagement trend",
    "hands_on_exercises_completed": "the hands-on exercises they completed",
    "project_submissions": "their project submissions",
    "forum_posts": "their forum participation",
    "goal_progress_score": "their progress toward their goal",
    "tenure_months": "how long they have subscribed",
    "payment_failures": "their payment failures",
    "support_tickets_raised": "the support tickets they raised",
    "satisfaction_score": "their satisfaction score",
    "plan_type_Monthly": "being on a monthly plan",
    "plan_type_Annual": "being on an annual plan",
}

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
    st.header("Predict: how likely is this learner to leave?")
    st.info("How to read this: describe a learner using the controls below. The risk updates instantly.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Learner profile")
        occupation = st.selectbox("Occupation", ["Student", "Working Professional", "Career Switcher"])
        goal = st.selectbox("Signup goal", ["Get a Job", "Certification", "Hobby"])
        plan = st.selectbox("Plan type", ["Monthly", "Annual"])
        discount = st.selectbox("Used a signup discount?", ["No", "Yes"])
        tenure = st.slider("Months subscribed", 0, 24, 6)
        payment_failures = st.slider("Payment failures", 0, 5, 0)
        support_tickets = st.slider("Support tickets raised", 0, 8, 0)
        satisfaction = st.slider("Satisfaction (1-5)", 1, 5, 3)

    with col2:
        st.subheader("Learning behaviour")
        completion = st.slider("Course completion (%)", 0, 100, 50)
        quiz = st.slider("Average quiz score", 0, 100, 65)
        replays = st.slider("Video replays", 0, 15, 3)
        days_login = st.slider("Days since last login", 0, 90, 7)
        trend_choice = st.selectbox("Engagement trend", ["Declining", "Flat", "Improving"], index=1)
        exercises = st.slider("Hands-on exercises completed", 0, 20, 4)
        projects = st.slider("Project submissions", 0, 8, 1)
        forum = st.slider("Forum posts", 0, 10, 1)
        goal_progress = st.slider("Progress toward their goal (0-100)", 0, 100, 40)

    trend_value = {"Declining": -1, "Flat": 0, "Improving": 1}[trend_choice]

    # Build one row that matches the model's expected columns exactly
    row = {c: 0 for c in model_columns}
    row.update({
        "avg_quiz_score": quiz,
        "video_replay_count": replays,
        "course_completion_rate": completion,
        "days_since_last_login": days_login,
        "engagement_trend": trend_value,
        "hands_on_exercises_completed": exercises,
        "project_submissions": projects,
        "forum_posts": forum,
        "goal_progress_score": goal_progress,
        "tenure_months": tenure,
        "payment_failures": payment_failures,
        "support_tickets_raised": support_tickets,
        "satisfaction_score": satisfaction,
    })
    for key in [f"occupation_type_{occupation}", f"signup_goal_{goal}",
                f"plan_type_{plan}", f"signup_discount_used_{discount}"]:
        if key in row:
            row[key] = 1

    input_df = pd.DataFrame([row], columns=model_columns)
    input_scaled = scaler.transform(input_df)
    risk = model.predict_proba(input_scaled)[0, 1]

    st.divider()
    st.subheader("Result")
    st.metric("Churn risk", f"{risk * 100:.1f}%")

    if risk < 0.3:
        st.success("🟢 Low risk: this learner is likely to stay. No action needed, a friendly 'keep it up' message is enough.")
    elif risk < 0.7:
        st.warning("🟡 Medium risk: this learner needs attention. Try a check-in email or a personalised course suggestion.")
    else:
        st.error("🔴 High risk: this learner is likely to leave. Act now with personal outreach or a win-back offer.")

    # Plain-English reasons (for a linear model, contribution = coefficient x scaled value)
    contrib = pd.Series(model.coef_[0] * input_scaled[0], index=model_columns)
    contrib = contrib[[c for c in contrib.index if c in FEATURE_LABELS]]
    risk_drivers = contrib[contrib > 0].sort_values(ascending=False).head(3)
    protectors = contrib[contrib < 0].sort_values().head(2)

    if len(risk_drivers) > 0:
        names = [FEATURE_LABELS[c] for c in risk_drivers.index]
        st.write("**Biggest reasons for risk:** " + ", ".join(names) + ".")
    if len(protectors) > 0:
        names = [FEATURE_LABELS[c] for c in protectors.index]
        st.write("**What is helping keep them:** " + ", ".join(names) + ".")

    st.caption("Note: the model is tuned to catch as many churners as possible, so its percentages run higher than real-world churn rates. Use them to rank learners, not as exact probabilities.")

# ============================== WHY? ==============================
elif page == "Why?":
    st.header("Why?")
    st.write("This page will show SHAP explanations.")

# ======================= RETENTION SIMULATOR =======================
elif page == "Retention Simulator":
    st.header("Retention Simulator")
    st.write("This page will show the ROI simulator.")
