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

# ---- Predict page: starting values and example learners ----
PRED_DEFAULTS = {
    "pr_occ": "Student", "pr_goal": "Get a Job", "pr_plan": "Monthly", "pr_disc": "No",
    "pr_tenure": 6, "pr_pf": 0, "pr_tickets": 0, "pr_sat": 3,
    "pr_comp": 50, "pr_quiz": 65, "pr_replays": 3, "pr_days": 7, "pr_trend": "Flat",
    "pr_ex": 4, "pr_proj": 1, "pr_forum": 1, "pr_gp": 40,
}

EXAMPLES = {
    "👻 The Ghost": {
        "story": "Signed up with a discount, barely started, and has not logged in for five weeks.",
        "values": {
            "pr_occ": "Student", "pr_goal": "Hobby", "pr_plan": "Monthly", "pr_disc": "Yes",
            "pr_tenure": 2, "pr_pf": 1, "pr_tickets": 0, "pr_sat": 2,
            "pr_comp": 15, "pr_quiz": 55, "pr_replays": 4, "pr_days": 35, "pr_trend": "Declining",
            "pr_ex": 0, "pr_proj": 0, "pr_forum": 0, "pr_gp": 15,
        },
    },
    "😣 The Struggler": {
        "story": "Keeps replaying videos, fails quizzes, raises support tickets, but is still trying.",
        "values": {
            "pr_occ": "Working Professional", "pr_goal": "Certification", "pr_plan": "Monthly", "pr_disc": "No",
            "pr_tenure": 4, "pr_pf": 0, "pr_tickets": 3, "pr_sat": 2,
            "pr_comp": 25, "pr_quiz": 45, "pr_replays": 9, "pr_days": 10, "pr_trend": "Declining",
            "pr_ex": 1, "pr_proj": 0, "pr_forum": 2, "pr_gp": 20,
        },
    },
    "🏆 The Champion": {
        "story": "Annual plan, finishing courses, doing projects, active in the forum, and happy.",
        "values": {
            "pr_occ": "Career Switcher", "pr_goal": "Get a Job", "pr_plan": "Annual", "pr_disc": "No",
            "pr_tenure": 14, "pr_pf": 0, "pr_tickets": 0, "pr_sat": 5,
            "pr_comp": 90, "pr_quiz": 88, "pr_replays": 1, "pr_days": 1, "pr_trend": "Improving",
            "pr_ex": 14, "pr_proj": 4, "pr_forum": 5, "pr_gp": 85,
        },
    },
    "🤫 The Quiet Achiever": {
        "story": "Great scores and completion, but has gone quiet for four weeks. Will the model worry?",
        "values": {
            "pr_occ": "Working Professional", "pr_goal": "Certification", "pr_plan": "Annual", "pr_disc": "No",
            "pr_tenure": 10, "pr_pf": 0, "pr_tickets": 0, "pr_sat": 4,
            "pr_comp": 88, "pr_quiz": 82, "pr_replays": 1, "pr_days": 28, "pr_trend": "Declining",
            "pr_ex": 9, "pr_proj": 2, "pr_forum": 1, "pr_gp": 60,
        },
    },
}

def load_example(name):
    for k, v in EXAMPLES[name]["values"].items():
        st.session_state[k] = v

def reset_learner():
    for k, v in PRED_DEFAULTS.items():
        st.session_state[k] = v

# ---- Sidebar: navigation + currency ----
page = st.sidebar.radio(
    "Navigate",
    ["Start Here", "Overview", "Explore", "Predict", "Why?", "Retention Simulator", "Try Your Own Data"],
)

currency_choice = st.sidebar.selectbox("Currency", ["₹ Indian Rupee", "$ US Dollar"])
IS_INR = currency_choice.startswith("₹")
SYM = "₹" if IS_INR else "$"
if IS_INR:
    DEF_MONTHLY, DEF_ANNUAL, DEF_OFFER, MAX_OFFER, MAX_PRICE = 800, 500, 150, 1000, 10000
else:
    DEF_MONTHLY, DEF_ANNUAL, DEF_OFFER, MAX_OFFER, MAX_PRICE = 15, 10, 8, 50, 200

# ---- Helper functions ----
def fmt_money(x):
    """Format money; uses Indian digit grouping (1,23,456) for rupees."""
    x = float(x)
    sign = "-" if x < 0 else ""
    n = int(round(abs(x)))
    s = str(n)
    if IS_INR and len(s) > 3:
        last3, rest = s[-3:], s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        s = ",".join(parts) + "," + last3
    elif not IS_INR:
        s = f"{n:,}"
    return f"{sign}{SYM}{s}"

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

st.title("📚 EdTech Learner Churn Dashboard")

# ========================== START HERE ==========================
if page == "Start Here":
    st.header("Start here: what is this project?")

    st.markdown(
        """
**The problem.** An online learning platform (think Coursera, Udemy or Coding Ninjas) earns money from
subscriptions. When a learner cancels, that is called **churn**. Finding out *after* someone leaves is too late.

**The idea.** This dashboard looks at how learners behave (logins, quizzes, videos, exercises, satisfaction)
and predicts **who is likely to leave, why, and what to do about it**, before they go.

**About the data.** The 3,000 learners here are *synthetic*: generated with realistic behaviour rules,
because real platforms do not publish their customer data. The patterns are realistic, but the learners are not real people.
        """
    )

    st.subheader("What each page does")
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
- **Overview:** the big numbers and the biggest patterns at a glance
- **Explore:** compare groups and behaviours, and see what separates learners who stay from those who leave
- **Predict:** describe any learner (or load an example) and get their risk (🟢 low, 🟡 medium, 🔴 high)
            """
        )
    with b:
        st.markdown(
            """
- **Why?:** what drives churn overall, and the reasons behind one learner's risk, plus their persona
- **Retention Simulator:** how much money a retention campaign could save, and a what-if tool for one learner
- **Try Your Own Data:** upload a CSV of learners and get every learner's risk, reason and persona
            """
        )

    st.subheader("🧪 Try these experiments")
    st.caption("Open each page from the sidebar and follow the steps. Each one takes about a minute.")

    with st.expander("Experiment 1: Spot the warning signs (Predict page)"):
        st.markdown(
            """
1. Open **Predict** and click an example learner at the top, like **The Champion**, then **The Ghost**. Compare the risk.
2. Click **Reset** and set **Engagement trend** to *Declining* and **Days since last login** to *30*. What happens to the risk?
3. Now set **Course completion** to *90* and **Satisfaction** to *5*. Does the risk recover?

*What you learn:* no single signal decides churn. Many small signals add up.
            """
        )

    with st.expander("Experiment 2: Which plan keeps learners longer? (Explore page)"):
        st.markdown(
            """
1. Open **Explore** and compare churn by **Plan type**.
2. Switch the comparison to **Signup goal**, then **Used a signup discount**.

*What you learn:* how much commitment (annual plans, clear goals) matters.
            """
        )

    with st.expander("Experiment 3: What can we do to save one learner? (Retention Simulator)"):
        st.markdown(
            """
1. Open **Retention Simulator** and scroll to **What-if**.
2. Click **Pick a random at-risk learner**.
3. Choose **Hands-on exercises completed** and slide it up. Then try **Days since last login**.

*What you learn:* which actions actually reduce risk, so a team knows where to spend effort.
            """
        )

    with st.expander("Experiment 4: Is a campaign worth the money? (Retention Simulator)"):
        st.markdown(
            """
1. In **Retention Simulator**, set the offer cost to the maximum and the success rate to 5%.
2. Watch the ROI turn negative. Now move the sliders back and target fewer learners.

*What you learn:* targeting the right learners is what makes a retention campaign pay off.
            """
        )

    with st.expander("Experiment 5: Score a whole list of learners (Try Your Own Data)"):
        st.markdown(
            """
1. Open **Try Your Own Data** and use the sample file. Look at the learners sorted by risk.
2. Download the CSV template, change a few values in a spreadsheet, and upload it back.

*What you learn:* how a retention team could get a weekly "who to contact first" list.
            """
        )

    st.subheader("📖 Glossary")
    glossary = {
        "Churn": "When a learner cancels or stops renewing their subscription.",
        "Churn rate": "The share of learners who left. A 23% churn rate means about 1 in 4 left.",
        "Risk level (🟢🟡🔴)": "Low risk is under 30%, medium is 30 to 70%, high is above 70% predicted chance of leaving.",
        "Engagement trend": "Whether a learner's activity is going up, staying flat or declining over recent weeks.",
        "Persona": "A named group of similar at-risk learners (for example 'Struggling but Trying'), each with a suggested action.",
        "Lifetime value": "The money a learner is expected to bring in over their whole subscription.",
        "ROI": "Return on investment: how much money a campaign saves compared with what it costs.",
        "Recall": "Of all learners who really left, how many the model caught. We tuned the model to catch as many as possible.",
    }
    for term, meaning in glossary.items():
        with st.expander(term):
            st.write(meaning)

    st.subheader("⚠️ Honest limits")
    st.markdown(
        """
- The data is synthetic, so results show how the method works, not how a real company behaves.
- The model is tuned to catch most churners, so it raises more false alarms. Use the risk to **rank** learners, not as an exact probability.
- Money amounts use assumed prices that you can change in the Retention Simulator.
        """
    )

# =========================== OVERVIEW ===========================
elif page == "Overview":
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
    st.info("How to read this: load an example learner or set the controls yourself. The risk updates instantly.")

    # Make sure every control has a starting value (only the first time)
    for k, v in PRED_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.subheader("Quick start: load an example learner")
    ex_cols = st.columns(len(EXAMPLES) + 1)
    for col, name in zip(ex_cols, EXAMPLES.keys()):
        col.button(name, on_click=load_example, args=(name,), use_container_width=True)
    ex_cols[-1].button("↺ Reset", on_click=reset_learner, use_container_width=True)

    with st.expander("Who are these example learners?"):
        for name, ex in EXAMPLES.items():
            st.write(f"**{name}:** {ex['story']}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Learner profile")
        occupation = st.selectbox("Occupation", ["Student", "Working Professional", "Career Switcher"], key="pr_occ")
        goal = st.selectbox("Signup goal", ["Get a Job", "Certification", "Hobby"], key="pr_goal")
        plan = st.selectbox("Plan type", ["Monthly", "Annual"], key="pr_plan")
        discount = st.selectbox("Used a signup discount?", ["No", "Yes"], key="pr_disc")
        tenure = st.slider("Months subscribed", 0, 24, key="pr_tenure")
        payment_failures = st.slider("Payment failures", 0, 5, key="pr_pf")
        support_tickets = st.slider("Support tickets raised", 0, 8, key="pr_tickets")
        satisfaction = st.slider("Satisfaction (1-5)", 1, 5, key="pr_sat")

    with col2:
        st.subheader("Learning behaviour")
        completion = st.slider("Course completion (%)", 0, 100, key="pr_comp")
        quiz = st.slider("Average quiz score", 0, 100, key="pr_quiz")
        replays = st.slider("Video replays", 0, 15, key="pr_replays")
        days_login = st.slider("Days since last login", 0, 90, key="pr_days")
        trend_choice = st.selectbox("Engagement trend", ["Declining", "Flat", "Improving"], key="pr_trend")
        exercises = st.slider("Hands-on exercises completed", 0, 20, key="pr_ex")
        projects = st.slider("Project submissions", 0, 8, key="pr_proj")
        forum = st.slider("Forum posts", 0, 10, key="pr_forum")
        goal_progress = st.slider("Progress toward their goal (0-100)", 0, 100, key="pr_gp")

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
    offer_cost = s1.slider(f"Cost per retention offer ({SYM})", 1, MAX_OFFER, DEF_OFFER, key=f"offer_{SYM}")
    success_rate = s2.slider("Offer success rate (%)", 5, 80, 35) / 100
    target_pct = s3.slider("Share of at-risk learners to target (%)", 5, 100, 20)

    p1, p2 = st.columns(2)
    monthly_price = p1.number_input(f"Monthly plan price ({SYM} per month)", min_value=1, max_value=MAX_PRICE,
                                    value=DEF_MONTHLY, key=f"mp_{SYM}")
    annual_price = p2.number_input(f"Annual plan price ({SYM} per month equivalent)", min_value=1, max_value=MAX_PRICE,
                                   value=DEF_ANNUAL, key=f"ap_{SYM}")

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
    k3.metric("Campaign cost", fmt_money(campaign_cost))
    k4.metric("ROI", f"{roi:,.0f}%")

    k5, k6 = st.columns(2)
    k5.metric("Expected revenue saved", fmt_money(revenue_saved))
    k6.metric("Net value (saved minus cost)", fmt_money(net_value))

    compare_chart = pd.DataFrame(
        {f"Amount ({SYM})": [do_nothing_loss, loss_with_campaign, campaign_cost]},
        index=["Revenue lost: do nothing", "Revenue lost: run campaign", "Campaign cost"],
    )
    st.bar_chart(compare_chart)

    if net_value > 0:
        st.success(
            f"Takeaway: spending {fmt_money(campaign_cost)} on {n_target} learners could protect about "
            f"{fmt_money(revenue_saved)} in revenue, compared with {fmt_money(do_nothing_loss)} at risk if the company does nothing."
        )
    else:
        st.warning(
            f"Takeaway: with these settings the campaign costs {fmt_money(campaign_cost)} but saves only "
            f"{fmt_money(revenue_saved)}. Try targeting fewer learners or a cheaper offer."
        )

    with st.expander("Who would we target first? (top 10)"):
        top10 = targeted.head(10).copy()
        show = pd.DataFrame({
            "Plan": top10['plan_type'],
            "Months subscribed": top10['tenure_months'],
            "Churn risk (%)": (top10['risk'] * 100).round(1),
            f"Lifetime value ({SYM})": top10['clv'].round(0),
        })
        st.dataframe(show, use_container_width=True)

    st.caption(
        "Simplified model: learners are ranked by churn risk x lifetime value, and every targeted learner "
        "is assumed to be worth saving. Prices are assumptions you can change. Real savings would be lower."
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

# ======================= TRY YOUR OWN DATA =======================
elif page == "Try Your Own Data":
    st.header("Try your own data: score a whole list of learners")
    st.info("How to read this: upload a CSV of learners (or use the sample file) and get each learner's risk level, biggest risk factor, persona and recommended action. You can download the results.")

    CAT_ALLOWED = {
        "occupation_type": ["Student", "Working Professional", "Career Switcher"],
        "signup_goal": ["Get a Job", "Certification", "Hobby"],
        "plan_type": ["Monthly", "Annual"],
        "signup_discount_used": ["Yes", "No"],
    }
    NUM_COLS = [
        "avg_quiz_score", "video_replay_count", "course_completion_rate", "days_since_last_login",
        "hands_on_exercises_completed", "project_submissions", "forum_posts", "goal_progress_score",
        "tenure_months", "payment_failures", "support_tickets_raised", "satisfaction_score",
    ]
    RAW_COLS = list(CAT_ALLOWED.keys()) + ["engagement_trend"] + NUM_COLS

    # A ready-made sample of 50 learners (and the template is its first 5 rows)
    sample_df = df.sample(50, random_state=3)[RAW_COLS].reset_index(drop=True)
    sample_df.insert(0, "learner_id", [f"L{i + 1:03d}" for i in range(len(sample_df))])
    template_df = sample_df.head(5)

    with st.expander("What columns does my file need?"):
        st.markdown(
            """
Your CSV needs one row per learner and these columns (column names are not case sensitive; an optional
`learner_id` column is kept so you can recognise each learner):

- **occupation_type:** Student, Working Professional or Career Switcher
- **signup_goal:** Get a Job, Certification or Hobby
- **plan_type:** Monthly or Annual
- **signup_discount_used:** Yes or No
- **engagement_trend:** Declining, Flat or Improving (or -1, 0, 1)
- **Numbers:** avg_quiz_score, video_replay_count, course_completion_rate, days_since_last_login,
  hands_on_exercises_completed, project_submissions, forum_posts, goal_progress_score, tenure_months,
  payment_failures, support_tickets_raised, satisfaction_score

Tip: download the template below, replace the rows with your own learners, and upload it.
            """
        )

    st.download_button(
        "⬇ Download CSV template",
        template_df.to_csv(index=False).encode("utf-8"),
        file_name="learner_template.csv",
        mime="text/csv",
    )

    source = st.radio("Choose your data", ["Use the sample file (50 learners)", "Upload my own CSV"], horizontal=True)

    raw = None
    if source.startswith("Use"):
        raw = sample_df.copy()
    else:
        up_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if up_file is not None:
            try:
                raw = pd.read_csv(up_file)
            except Exception:
                st.error("Sorry, that file could not be read as a CSV. Try downloading the template and starting from it.")

    if raw is None:
        st.stop()

    if len(raw) > 5000:
        st.warning("Your file has more than 5,000 rows, so only the first 5,000 are scored.")
        raw = raw.head(5000)

    # Make column names forgiving: "Plan Type" and "plan_type" both work
    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]

    missing = [c for c in RAW_COLS if c not in raw.columns]
    if missing:
        st.error("Your file is missing these columns: " + ", ".join(missing) + ". Download the template to see the format.")
        st.stop()

    ids = raw["learner_id"].astype(str) if "learner_id" in raw.columns else pd.Series(range(1, len(raw) + 1), index=raw.index).astype(str)

    clean = raw[RAW_COLS].copy()

    # Text columns: tidy up spelling/case, mark unknown values as missing
    for col, allowed in CAT_ALLOWED.items():
        canon = {v.lower(): v for v in allowed}
        clean[col] = clean[col].astype(str).str.strip().str.lower().map(canon)

    # Engagement trend: accept -1/0/1 or the words
    trend_raw = raw["engagement_trend"]
    trend_num = pd.to_numeric(trend_raw, errors="coerce")
    trend_txt = trend_raw.astype(str).str.strip().str.lower().map({"declining": -1, "flat": 0, "improving": 1})
    clean["engagement_trend"] = trend_num.where(trend_num.notna(), trend_txt)
    clean.loc[~clean["engagement_trend"].isin([-1, 0, 1]), "engagement_trend"] = np.nan

    # Number columns
    for col in NUM_COLS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    # Drop rows with anything missing or invalid
    bad_rows = clean.isna().any(axis=1)
    n_bad = int(bad_rows.sum())
    if n_bad > 0:
        st.warning(f"{n_bad} row(s) had a missing or invalid value and were skipped.")
    clean = clean[~bad_rows].copy()
    ids_clean = ids[~bad_rows]

    if len(clean) == 0:
        st.error("None of the rows could be scored. Check the allowed values and try again.")
        st.stop()

    # Keep numbers inside the range the model was trained on
    n_clipped = 0
    for col in NUM_COLS:
        lo, hi = df[col].min(), df[col].max()
        clipped = clean[col].clip(lo, hi)
        n_clipped += int((clipped != clean[col]).sum())
        clean[col] = clipped
    if n_clipped > 0:
        st.caption(f"{n_clipped} value(s) were outside the range the model was trained on and were limited to that range.")

    # ---- Score every learner ----
    X_up = pd.get_dummies(clean[RAW_COLS], columns=list(CAT_ALLOWED.keys()))
    X_up = X_up.reindex(columns=model_columns, fill_value=0).astype(float)
    X_scaled = scaler.transform(X_up)
    risk_up = model.predict_proba(X_scaled)[:, 1]

    # Biggest risk factor per learner (linear model: coefficient x scaled value)
    contrib = pd.DataFrame(X_scaled * model.coef_[0], columns=model_columns)
    contrib = contrib[[c for c in model_columns if c in SHORT_LABELS]]
    top_col = contrib.idxmax(axis=1)
    top_val = contrib.max(axis=1)

    # Persona (peer percentile is measured against the original 3,000 learners)
    ref_sorted = np.sort(df["course_completion_rate"].values)
    peer = np.searchsorted(ref_sorted, clean["course_completion_rate"].values, side="right") / len(ref_sorted) * 100
    cl_in = clean[[c for c in cluster_columns if c != "peer_completion_percentile"]].copy()
    cl_in["peer_completion_percentile"] = peer
    cl_in = cl_in[cluster_columns].astype(float)
    cluster_ids = kmeans.predict(cluster_scaler.transform(cl_in))

    levels, reasons, personas, actions = [], [], [], []
    for r, c, v, cid in zip(risk_up, top_col, top_val, cluster_ids):
        levels.append(tier_label(r))
        if r >= 0.3:
            reasons.append(SHORT_LABELS[c] if v > 0 else "-")
            info = persona_info[str(int(cid))]
            personas.append(info["name"])
            actions.append(info["action"])
        else:
            reasons.append("-")
            personas.append("-")
            actions.append("No action needed")

    results = pd.DataFrame({
        "Learner": ids_clean.values,
        "Churn risk (%)": (risk_up * 100).round(1),
        "Risk level": levels,
        "Biggest risk factor": reasons,
        "Persona": personas,
        "Recommended action": actions,
    }).sort_values("Churn risk (%)", ascending=False).reset_index(drop=True)

    # ---- Show the results ----
    st.divider()
    st.subheader("Results")

    order = ["🟢 Low", "🟡 Medium", "🔴 High"]
    counts = results["Risk level"].value_counts().reindex(order, fill_value=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Learners scored", f"{len(results):,}")
    m2.metric("🔴 High risk", f"{int(counts['🔴 High']):,}")
    m3.metric("🟡 Medium risk", f"{int(counts['🟡 Medium']):,}")
    m4.metric("🟢 Low risk", f"{int(counts['🟢 Low']):,}")

    st.bar_chart(counts)

    n_act = int(counts["🔴 High"] + counts["🟡 Medium"])
    st.success(
        f"Takeaway: {n_act} of {len(results)} learners ({n_act / len(results) * 100:.0f}%) need attention. "
        f"The table below is sorted so the learners to contact first are at the top."
    )

    st.dataframe(results, use_container_width=True)

    st.download_button(
        "⬇ Download results as CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="churn_risk_results.csv",
        mime="text/csv",
    )

    st.caption(
        "Privacy: this app does not save uploaded files. The model was trained on synthetic data, so results show how "
        "the method works and should not be used for real decisions. Please do not upload real personal data."
    )
