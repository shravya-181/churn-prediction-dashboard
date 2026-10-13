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

# Short names for charts
SHORT_LABELS = {
    "avg_quiz_score": "Quiz score",
    "video_replay_count": "Video replays",
    "course_completion_rate": "Course completion",
    "days_since_last_login": "Days since last login",
    "engagement_trend": "Engagement trend",
    "hands_on_exercises_completed": "Hands-on exercises",
    "project_submissions": "Project submissions",
    "forum_posts": "Forum posts",
    "goal_progress_score": "Goal progress",
    "tenure_months": "Months subscribed",
    "payment_failures": "Payment failures",
    "support_tickets_raised": "Support tickets",
    "satisfaction_score": "Satisfaction",
    "plan_type_Monthly": "Monthly plan (vs annual)",
}

# ---- Helper functions ----
def tier_label(r):
    if r < 0.3:
        return "🟢 Low"
    elif r < 0.7:
        return "🟡 Medium"
    return "🔴 High"

def learner_risk(learner, overrides=None):
    """Churn risk (0-1) for one learner row from df, with optional changed values."""
    overrides = overrides or {}
    row = {c: 0 for c in model_columns}
    for c in model_columns:
        if c in learner.index:
            row[c] = learner[c]
    for key in [f"occupation_type_{learner['occupation_type']}",
                f"signup_goal_{learner['signup_goal']}",
                f"plan_type_{learner['plan_type']}",
                f"signup_discount_used_{learner['signup_discount_used']}"]:
        if key in row:
            row[key] = 1
    for c, v in overrides.items():
        row[c] = v
    input_df = pd.DataFrame([row], columns=model_columns).astype(float)
    return model.predict_proba(scaler.transform(input_df))[0, 1]

@st.cache_data
def score_all_learners():
    """Churn risk (0-1) for every learner in the dataset."""
    X_all = pd.get_dummies(
        df.drop(columns=['churned', 'days_to_churn']),
        columns=['occupation_type', 'signup_goal', 'plan_type', 'signup_discount_used'],
    )
    X_all = X_all.reindex(columns=model_columns, fill_value=0).astype(float)
    return model.predict_proba(scaler.transform(X_all))[:, 1]

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
    st.header("Why do learners leave?")
    st.info("How to read this: the first chart shows what matters most across ALL learners. The second explains ONE learner.")

    # ---- Part 1: overall drivers ----
    st.subheader("1. The biggest drivers of churn overall")
    coefs = pd.Series(model.coef_[0], index=model_columns)
    coefs = coefs[[c for c in coefs.index if c in SHORT_LABELS]]
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10)
    chart = pd.DataFrame(
        {"Effect on churn risk (above 0 = raises risk, below 0 = lowers risk)": top.values},
        index=[SHORT_LABELS[c] for c in top.index],
    )
    st.bar_chart(chart)

    top_name = SHORT_LABELS[top.index[0]]
    top_dir = "lowers" if top.iloc[0] < 0 else "raises"
    risk_names = [SHORT_LABELS[c] for c in top.index if top[c] > 0][:3]
    st.success(
        f"Takeaway: '{top_name}' matters most, and a higher value {top_dir} churn risk. "
        f"The strongest warning signs are: {', '.join(risk_names)}."
    )

    # ---- Part 2: explain one learner ----
    st.divider()
    st.subheader("2. Explain one learner")

    if "learner_idx" not in st.session_state:
        st.session_state["learner_idx"] = 0

    b1, b2 = st.columns([1, 3])
    if b1.button("Pick a random learner"):
        st.session_state["learner_idx"] = int(np.random.randint(0, len(df)))
    idx = b2.number_input(f"Or enter a learner number (0 to {len(df) - 1})",
                          min_value=0, max_value=len(df) - 1, step=1, key="learner_idx")

    learner = df.iloc[int(idx)]

    # Build the model row for this learner
    row = {c: 0 for c in model_columns}
    for c in model_columns:
        if c in learner.index:
            row[c] = learner[c]
    for key in [f"occupation_type_{learner['occupation_type']}",
                f"signup_goal_{learner['signup_goal']}",
                f"plan_type_{learner['plan_type']}",
                f"signup_discount_used_{learner['signup_discount_used']}"]:
        if key in row:
            row[key] = 1

    input_df = pd.DataFrame([row], columns=model_columns).astype(float)
    input_scaled = scaler.transform(input_df)
    risk = model.predict_proba(input_scaled)[0, 1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Churn risk", f"{risk * 100:.1f}%")
    m2.metric("Course completion", f"{learner['course_completion_rate']:.0f}%")
    m3.metric("Days since last login", f"{int(learner['days_since_last_login'])}")
    m4.metric("Actually churned?", "Yes" if learner['churned'] == 1 else "No")

    if risk < 0.3:
        st.success("🟢 Low risk")
    elif risk < 0.7:
        st.warning("🟡 Medium risk")
    else:
        st.error("🔴 High risk")

    # Reasons for this learner (linear model: contribution = coefficient x scaled value)
    contrib = pd.Series(model.coef_[0] * input_scaled[0], index=model_columns)
    contrib = contrib[[c for c in contrib.index if c in SHORT_LABELS]]
    top_contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
    contrib_chart = pd.DataFrame(
        {"Push on risk (above 0 = toward leaving, below 0 = toward staying)": top_contrib.values},
        index=[SHORT_LABELS[c] for c in top_contrib.index],
    )
    st.bar_chart(contrib_chart)

    up = [FEATURE_LABELS.get(c, c) for c in contrib[contrib > 0].sort_values(ascending=False).head(3).index]
    down = [FEATURE_LABELS.get(c, c) for c in contrib[contrib < 0].sort_values().head(2).index]
    summary = f"This learner has a {risk * 100:.0f}% churn risk."
    if up:
        summary += f" The biggest reasons are {', '.join(up)}."
    if down:
        summary += f" Helping keep them: {', '.join(down)}."
    st.write(summary)

    # Persona (personas were built for at-risk learners, so only show for medium/high risk)
    if risk >= 0.3:
        cluster_input = pd.DataFrame([learner[cluster_columns].astype(float)])
        cluster_id = int(kmeans.predict(cluster_scaler.transform(cluster_input))[0])
        info = persona_info[str(cluster_id)]
        st.subheader(f"Persona: {info['name']}")
        st.write(f"**Recommended action:** {info['action']}")
    else:
        st.caption("Personas only apply to learners at medium or high risk.")

    st.caption("Note: the model is tuned to catch as many churners as possible, so its percentages run higher than real-world churn rates. Use them to rank learners, not as exact probabilities.")

# ======================= RETENTION SIMULATOR =======================
elif page == "Retention Simulator":
    st.header("Retention Simulator: is a retention campaign worth it?")
    st.info("How to read this: set the campaign assumptions, and see how much money a targeted retention campaign could protect.")

    risk_all = score_all_learners()

    # ---------------- Part 1: Campaign ROI ----------------
    st.subheader("1. Campaign ROI")

    s1, s2, s3 = st.columns(3)
    offer_cost = s1.slider("Cost per retention offer ($)", 1, 50, 8)
    success_rate = s2.slider("Offer success rate (%)", 5, 80, 35) / 100
    target_pct = s3.slider("Share of at-risk learners to target (%)", 5, 100, 20)

    p1, p2 = st.columns(2)
    monthly_price = p1.number_input("Monthly plan price ($ per month)", min_value=1, max_value=100, value=15)
    annual_price = p2.number_input("Annual plan price ($ per month equivalent)", min_value=1, max_value=100, value=10)

    biz = df[['plan_type', 'tenure_months']].copy()
    biz['risk'] = risk_all
    biz['price'] = biz['plan_type'].map({'Monthly': monthly_price, 'Annual': annual_price})
    biz['clv'] = biz['price'] * (biz['tenure_months'] + 1)

    at_risk_df = biz[biz['risk'] >= 0.3].copy()

    if len(at_risk_df) == 0:
        st.warning("No at-risk learners found.")
        st.stop()

    # Prioritise by risk x value: save the most valuable, most likely leavers first
    at_risk_df['priority'] = at_risk_df['risk'] * at_risk_df['clv']
    at_risk_df = at_risk_df.sort_values('priority', ascending=False)

    n_target = max(1, int(len(at_risk_df) * target_pct / 100))
    targeted = at_risk_df.head(n_target)

    campaign_cost = n_target * offer_cost
    revenue_saved = (targeted['clv'] * success_rate).sum()
    net_value = revenue_saved - campaign_cost
    roi = net_value / campaign_cost * 100
    do_nothing_loss = (at_risk_df['clv'] * at_risk_df['risk']).sum()
    loss_with_campaign = max(do_nothing_loss - revenue_saved, 0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("At-risk learners", f"{len(at_risk_df):,}")
    k2.metric("Learners targeted", f"{n_target:,}")
    k3.metric("Campaign cost", f"${campaign_cost:,.0f}")
    k4.metric("ROI", f"{roi:,.0f}%")

    k5, k6 = st.columns(2)
    k5.metric("Expected revenue saved", f"${revenue_saved:,.0f}")
    k6.metric("Net value (saved minus cost)", f"${net_value:,.0f}")

    compare_chart = pd.DataFrame(
        {"Dollars ($)": [do_nothing_loss, loss_with_campaign, campaign_cost]},
        index=["Revenue lost: do nothing", "Revenue lost: run campaign", "Campaign cost"],
    )
    st.bar_chart(compare_chart)

    if net_value > 0:
        st.success(
            f"Takeaway: spending ${campaign_cost:,.0f} on {n_target} learners could protect about "
            f"${revenue_saved:,.0f} in revenue, compared with ${do_nothing_loss:,.0f} at risk if the company does nothing."
        )
    else:
        st.warning(
            f"Takeaway: with these settings the campaign costs ${campaign_cost:,.0f} but saves only "
            f"${revenue_saved:,.0f}. Try targeting fewer learners or a cheaper offer."
        )

    with st.expander("Who would we target first? (top 10)"):
        top10 = targeted.head(10).copy()
        show = pd.DataFrame({
            "Plan": top10['plan_type'],
            "Months subscribed": top10['tenure_months'],
            "Churn risk (%)": (top10['risk'] * 100).round(1),
            "Lifetime value ($)": top10['clv'].round(0),
        })
        st.dataframe(show, use_container_width=True)

    st.caption(
        "Simplified model: learners are ranked by churn risk x lifetime value, and every targeted learner "
        "is assumed to be worth saving. Real savings would be lower."
    )

    # ---------------- Part 2: What-if ----------------
    st.divider()
    st.subheader("2. What-if: what if we help one learner improve?")
    st.info("How to read this: pick a learner, change one thing, and see how their churn risk changes.")

    pool = np.where(risk_all >= 0.3)[0]
    start_pool = np.where(risk_all >= 0.5)[0]

    if "wi_idx" not in st.session_state:
        st.session_state["wi_idx"] = int(start_pool[0]) if len(start_pool) else 0

    w1, w2 = st.columns([1, 3])
    if w1.button("Pick a random at-risk learner", key="wi_random"):
        st.session_state["wi_idx"] = int(np.random.choice(pool))
    wi_idx = int(w2.number_input(f"Or enter a learner number (0 to {len(df) - 1})",
                                 min_value=0, max_value=len(df) - 1, step=1, key="wi_idx"))

    learner = df.iloc[wi_idx]

    FACTORS = {
        "Course completion (%)": ("course_completion_rate", 0, 100),
        "Hands-on exercises completed": ("hands_on_exercises_completed", 0, 20),
        "Days since last login": ("days_since_last_login", 0, 90),
        "Satisfaction (1-5)": ("satisfaction_score", 1, 5),
        "Project submissions": ("project_submissions", 0, 8),
        "Forum posts": ("forum_posts", 0, 10),
        "Support tickets raised": ("support_tickets_raised", 0, 8),
    }
    factor_label = st.selectbox("Change this:", list(FACTORS.keys()), key="wi_factor")
    col_name, lo, hi = FACTORS[factor_label]
    current = int(np.clip(round(float(learner[col_name])), lo, hi))
    new_val = st.slider(f"New value (currently {current})", lo, hi, current,
                        key=f"wi_{col_name}_{wi_idx}")

    before = learner_risk(learner, {col_name: current})
    after = learner_risk(learner, {col_name: new_val})

    r1, r2, r3 = st.columns(3)
    r1.metric("Risk before", f"{before * 100:.1f}%")
    r2.metric("Risk after", f"{after * 100:.1f}%",
              delta=f"{(after - before) * 100:+.1f} points", delta_color="inverse")
    r3.metric("Risk level", f"{tier_label(before)} → {tier_label(after)}")

    if after < before:
        st.success(f"Takeaway: changing '{factor_label}' from {current} to {new_val} lowers this learner's risk by {(before - after) * 100:.1f} points.")
    elif after > before:
        st.warning(f"Takeaway: changing '{factor_label}' from {current} to {new_val} raises this learner's risk by {(after - before) * 100:.1f} points.")
    else:
        st.info("Move the slider to see how the risk changes.")

    st.caption("Only one factor changes at a time. In real life, factors move together (a learner who completes more courses usually also does more exercises).")
