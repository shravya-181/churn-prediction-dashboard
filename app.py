import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="EdTech Churn Dashboard", layout="wide")

# ============================================================
# LOAD EVERYTHING ONCE, AT STARTUP
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('churn_model.pkl')          # log_model_v2
    scaler = joblib.load('churn_scaler.pkl')         # scaler_v2
    kmeans = joblib.load('persona_kmeans.pkl')       # kmeans_final
    cluster_scaler = joblib.load('persona_scaler.pkl')
    with open('model_columns.json') as f:
        model_columns = json.load(f)
    with open('cluster_columns.json') as f:
        cluster_columns = json.load(f)
    with open('persona_info.json') as f:
        persona_info = json.load(f)
    df = pd.read_csv('edtech_churn_dataset.csv')

    # ---- Predict churn probability for every learner ----
    model_input = pd.get_dummies(
        df.copy(),
        columns=['occupation_type', 'signup_goal', 'plan_type', 'signup_discount_used']
    )
    for col in model_columns:
        if col not in model_input.columns:
            model_input[col] = 0
    model_input = model_input[model_columns].astype(float)
    model_input_scaled = scaler.transform(model_input)
    df['churn_probability'] = model.predict_proba(model_input_scaled)[:, 1]

    # ---- Assign personas to at-risk learners only (matches how clustering was trained) ----
    df['persona'] = 'Not at risk'
    at_risk_mask = df['churn_probability'] > 0.3
    at_risk_cluster_data = df.loc[at_risk_mask, cluster_columns]
    at_risk_cluster_scaled = cluster_scaler.transform(at_risk_cluster_data)
    at_risk_clusters = kmeans.predict(at_risk_cluster_scaled)
    df.loc[at_risk_mask, 'persona'] = [persona_info[str(c)]['name'] for c in at_risk_clusters]
    df.loc[at_risk_mask, 'recommended_action'] = [persona_info[str(c)]['action'] for c in at_risk_clusters]

    return model, scaler, kmeans, cluster_scaler, model_columns, cluster_columns, persona_info, df


model, scaler, kmeans, cluster_scaler, model_columns, cluster_columns, persona_info, df = load_artifacts()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
page = st.sidebar.radio("Navigate", ["Overview", "Explore", "Predict", "Why?", "Retention Simulator"])
st.title("📚 EdTech Learner Churn Dashboard")

# ============================================================
# PAGE 1: OVERVIEW
# ============================================================
if page == "Overview":
    st.header("Overview")

    total_learners = len(df)
    churn_rate = df['churned'].mean() * 100
    total_revenue_at_risk = (df['churned'] * df['clv']).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Learners", f"{total_learners:,}")
    col2.metric("Historical Churn Rate", f"{churn_rate:.1f}%")
    col3.metric("Revenue Lost to Churn (historical)", f"${total_revenue_at_risk:,.0f}")

    st.markdown("---")
    st.subheader("Current At-Risk Learners by Persona")

    persona_summary = (
        df[df['persona'] != 'Not at risk']
        .groupby('persona')['churn_probability']
        .agg(['count', 'mean'])
        .rename(columns={'count': 'Learners', 'mean': 'Avg Risk'})
        .sort_values('Learners', ascending=False)
    )
    persona_summary['Avg Risk'] = (persona_summary['Avg Risk'] * 100).round(1).astype(str) + '%'
    st.dataframe(persona_summary, use_container_width=True)

# ============================================================
# PAGE 2: EXPLORE
# ============================================================
elif page == "Explore":
    st.header("Explore")

    st.subheader("Churn Rate by Plan Type")
    st.bar_chart(df.groupby('plan_type')['churned'].mean())

    st.subheader("Churn Rate by Engagement Trend")
    trend_labels = {-1: 'Declining', 0: 'Flat', 1: 'Improving'}
    trend_data = df.groupby('engagement_trend')['churned'].mean()
    trend_data.index = trend_data.index.map(trend_labels)
    st.bar_chart(trend_data)

    st.subheader("Course Completion Rate: Churned vs Stayed")
    col1, col2 = st.columns(2)
    col1.metric("Stayed — Avg Completion", f"{df[df['churned']==0]['course_completion_rate'].mean():.1f}%")
    col2.metric("Churned — Avg Completion", f"{df[df['churned']==1]['course_completion_rate'].mean():.1f}%")

    st.subheader("Distribution of Current Churn Risk Scores")
    st.bar_chart(np.histogram(df['churn_probability'], bins=10)[0])

# ============================================================
# PAGE 3: PREDICT
# ============================================================
elif page == "Predict":
    st.header("Predict a Learner's Churn Risk")
    st.write("Enter a learner's details to get their churn risk score.")

    col1, col2, col3 = st.columns(3)
    with col1:
        occupation_type = st.selectbox("Occupation Type", ['Student', 'Working Professional', 'Career Switcher'])
        signup_goal = st.selectbox("Signup Goal", ['Certification', 'Get a Job', 'Hobby'])
        plan_type = st.selectbox("Plan Type", ['Monthly', 'Annual'])
        signup_discount_used = st.selectbox("Used Signup Discount?", ['Yes', 'No'])
    with col2:
        avg_quiz_score = st.slider("Avg Quiz Score", 0, 100, 70)
        video_replay_count = st.slider("Video Replay Count", 0, 20, 2)
        course_completion_rate = st.slider("Course Completion Rate (%)", 0, 100, 60)
        days_since_last_login = st.slider("Days Since Last Login", 0, 90, 7)
        engagement_trend = st.selectbox("Engagement Trend", [-1, 0, 1],
                                         format_func=lambda x: {-1: 'Declining', 0: 'Flat', 1: 'Improving'}[x])
    with col3:
        hands_on_exercises_completed = st.slider("Hands-on Exercises Completed", 0, 20, 5)
        project_submissions = st.slider("Project Submissions", 0, 10, 1)
        forum_posts = st.slider("Forum Posts", 0, 10, 1)
        goal_progress_score = st.slider("Goal Progress Score", 0, 100, 50)
        peer_completion_percentile = st.slider("Peer Completion Percentile", 0, 100, 50)
        tenure_months = st.slider("Tenure (months)", 0, 24, 6)
        payment_failures = st.slider("Payment Failures", 0, 4, 0)
        support_tickets_raised = st.slider("Support Tickets Raised", 0, 6, 0)
        satisfaction_score = st.slider("Satisfaction Score (1-5)", 1, 5, 4)

    if st.button("Get Risk Score"):
        raw_input = pd.DataFrame([{
            'occupation_type': occupation_type,
            'signup_goal': signup_goal,
            'plan_type': plan_type,
            'signup_discount_used': signup_discount_used,
            'avg_quiz_score': avg_quiz_score,
            'video_replay_count': video_replay_count,
            'course_completion_rate': course_completion_rate,
            'days_since_last_login': days_since_last_login,
            'engagement_trend': engagement_trend,
            'hands_on_exercises_completed': hands_on_exercises_completed,
            'project_submissions': project_submissions,
            'forum_posts': forum_posts,
            'goal_progress_score': goal_progress_score,
            'peer_completion_percentile': peer_completion_percentile,
            'tenure_months': tenure_months,
            'payment_failures': payment_failures,
            'support_tickets_raised': support_tickets_raised,
            'satisfaction_score': satisfaction_score,
        }])

        encoded = pd.get_dummies(raw_input, columns=['occupation_type', 'signup_goal', 'plan_type', 'signup_discount_used'])
        for col in model_columns:
            if col not in encoded.columns:
                encoded[col] = 0
        encoded = encoded[model_columns].astype(float)
        scaled = scaler.transform(encoded)
        risk = model.predict_proba(scaled)[:, 1][0]

        st.markdown("---")
        st.metric("Churn Risk", f"{risk*100:.1f}%")

        if risk > 0.3:
            cluster_input = raw_input[cluster_columns]
            cluster_scaled = cluster_scaler.transform(cluster_input)
            cluster_id = kmeans.predict(cluster_scaled)[0]
            persona = persona_info[str(cluster_id)]
            st.warning(f"**At risk** — Persona: **{persona['name']}**")
            st.write(f"Recommended action: {persona['action']}")
        else:
            st.success("Low risk — no immediate action needed.")

# ============================================================
# PAGE 4: WHY?
# ============================================================
elif page == "Why?":
    st.header("Why? — What Drives Churn Risk")
    st.write(
        "These are the factors our model weighs most heavily, based on its learned coefficients. "
        "A positive value pushes churn risk **up**; a negative value pushes it **down**."
    )

    coef_df = pd.DataFrame({
        'feature': model_columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', ascending=False)

    st.subheader("Top factors increasing churn risk")
    st.dataframe(coef_df.head(8), use_container_width=True)

    st.subheader("Top factors decreasing churn risk")
    st.dataframe(coef_df.tail(8).sort_values('coefficient'), use_container_width=True)

    st.markdown("---")
    st.subheader("Explain a specific learner")
    learner_idx = st.number_input("Learner row number (0 to {})".format(len(df)-1), 0, len(df)-1, 0)
    st.write(df.iloc[[learner_idx]][['occupation_type', 'plan_type', 'churn_probability', 'persona']])

# ============================================================
# PAGE 5: RETENTION SIMULATOR
# ============================================================
elif page == "Retention Simulator":
    st.header("Retention Campaign ROI Simulator")

    offer_cost = st.slider("Cost per retention offer ($)", 1, 50, 8)
    offer_success_rate = st.slider("Offer success rate (%)", 5, 90, 35) / 100
    target_pct = st.slider("Target top X% highest-priority at-risk learners", 5, 100, 20) / 100

    at_risk = df[df['persona'] != 'Not at risk'].copy()
    at_risk['priority_score'] = at_risk['churn_probability'] * at_risk['clv']
    at_risk = at_risk.sort_values('priority_score', ascending=False)

    n_targeted = max(1, int(len(at_risk) * target_pct))
    targeted = at_risk.head(n_targeted)

    total_cost = n_targeted * offer_cost
    revenue_saved = (targeted['clv'] * offer_success_rate).sum()
    net_value = revenue_saved - total_cost
    do_nothing_loss = (at_risk['clv'] * at_risk['churn_probability']).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Learners Targeted", f"{n_targeted}")
    col2.metric("Campaign Cost", f"${total_cost:,.0f}")
    col3.metric("Expected Revenue Saved", f"${revenue_saved:,.0f}")

    col4, col5 = st.columns(2)
    col4.metric("Net Value", f"${net_value:,.0f}")
    col5.metric("ROI", f"{(net_value/total_cost)*100:.0f}%" if total_cost > 0 else "N/A")

    st.markdown("---")
    st.write(f"If you do nothing, expected revenue lost across all at-risk learners: **${do_nothing_loss:,.0f}**")
    st.write(f"Running this campaign protects **${revenue_saved:,.0f}** of that, at a cost of **{(total_cost/revenue_saved)*100:.1f}%** of revenue protected." if revenue_saved > 0 else "")
