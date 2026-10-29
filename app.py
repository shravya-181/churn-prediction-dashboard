import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="EdTech Churn Dashboard", page_icon="📚", layout="wide")

# ====================== LOOK AND FEEL (CSS) ======================
st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

[data-testid="stSidebar"] {background: linear-gradient(180deg, #F4F2FF 0%, #E6E0FF 100%);}
.side-brand {padding: 6px 4px 14px 4px; border-bottom: 2px solid #6C5CE7; margin-bottom: 12px;
             font-size: 1.15rem; color: #2D1B69;}
.side-brand span {font-size: 0.8rem; color: #6B6B8D;}

.hero {background: linear-gradient(120deg, #6C5CE7 0%, #8E7CF8 55%, #5AA9F0 100%);
       padding: 26px 32px; border-radius: 18px; margin-bottom: 22px;
       box-shadow: 0 6px 20px rgba(108, 92, 231, 0.25);}
.hero-title {font-size: 2rem; font-weight: 800; color: #FFFFFF; line-height: 1.2;}
.hero-sub {font-size: 1.02rem; color: rgba(255, 255, 255, 0.92); margin-top: 6px;}

h2, h3 {color: #2D1B69 !important;}

[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E6E1FA; border-left: 5px solid #6C5CE7;
                          padding: 14px 16px; border-radius: 12px; box-shadow: 0 2px 8px rgba(108, 92, 231, 0.08);}
[data-testid="stMetricValue"] {color: #2D1B69;}

[data-testid="stAlert"] {border-radius: 12px;}
[data-testid="stExpander"] {border: 1px solid #E6E1FA !important; border-radius: 12px; background: #FFFFFF;}

.stButton > button, [data-testid="stDownloadButton"] > button {
    border-radius: 10px; border: 1.5px solid #6C5CE7; font-weight: 600; color: #4B2FB5; background: #FFFFFF;}
.stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
    background: #6C5CE7; color: #FFFFFF; border-color: #6C5CE7;}

.feature-card {background: #FFFFFF; border: 1px solid #E6E1FA; border-radius: 16px; padding: 20px 18px;
               text-align: center; box-shadow: 0 3px 12px rgba(108, 92, 231, 0.10); height: 100%;}
.fc-icon {font-size: 2.2rem;}
.fc-title {font-size: 1.15rem; font-weight: 700; color: #2D1B69; margin: 6px 0 4px 0;}
.fc-text {font-size: 0.92rem; color: #4A4A68;}

.risk-banner {border-radius: 14px; padding: 16px 20px; margin: 8px 0 14px 0;}
.risk-title {font-size: 1.35rem; font-weight: 800;}
.risk-text {font-size: 0.98rem; margin-top: 4px; color: #2B2B3C;}
.risk-low {background: #E3F9F1; border-left: 8px solid #00B894;}
.risk-low .risk-title {color: #00785F;}
.risk-mid {background: #FFF6DB; border-left: 8px solid #FDCB6E;}
.risk-mid .risk-title {color: #9A6B00;}
.risk-high {background: #FFE6E0; border-left: 8px solid #E17055;}
.risk-high .risk-title {color: #B03A1F;}

.persona-card {background: #F4F2FF; border: 1px solid #D8D0FA; border-left: 8px solid #6C5CE7;
               border-radius: 14px; padding: 16px 20px; margin: 8px 0 14px 0;}
.persona-name {font-size: 1.25rem; font-weight: 800; color: #2D1B69;}
.persona-action {font-size: 0.97rem; margin-top: 4px; color: #2B2B3C;}

.app-footer {text-align: center; color: #8A8AA3; font-size: 0.85rem; margin-top: 40px;
             padding-top: 14px; border-top: 1px solid #E6E1FA;}
</style>
    """,
    unsafe_allow_html=True,
)

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

PALETTE = ["#6C5CE7", "#A29BFE", "#74B9FF", "#81ECEC"]
PERSONA_ICONS = {
    "Passive Coaster": "🛋️",
    "Goal-Focused Drifter": "🎯",
    "Quiet Achiever": "🤫",
    "Struggling but Trying": "😣",
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

# ---- Sidebar: branding + navigation + currency ----
st.sidebar.markdown(
    "<div class='side-brand'>📚 <b>Churn Dashboard</b><br><span>Predict · Explain · Act</span></div>",
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigate",
    ["Start Here", "Overview", "Explore", "Predict", "Why?", "Model Performance",
     "Survival Curves", "Retention Simulator", "Try Your Own Data"],
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

def hero(title, subtitle):
    st.markdown(
        f"<div class='hero'><div class='hero-title'>{title}</div><div class='hero-sub'>{subtitle}</div></div>",
        unsafe_allow_html=True,
    )

def risk_banner(risk, long=True):
    if risk < 0.3:
        cls, title = "risk-low", "🟢 Low risk"
        text = "This learner is likely to stay. No action needed, a friendly 'keep it up' message is enough."
    elif risk < 0.7:
        cls, title = "risk-mid", "🟡 Medium risk"
        text = "This learner needs attention. Try a check-in email or a personalised course suggestion."
    else:
        cls, title = "risk-high", "🔴 High risk"
        text = "This learner is likely to leave. Act now with personal outreach or a win-back offer."
    body = f"<div class='risk-text'>{text}</div>" if long else ""
    st.markdown(
        f"<div class='risk-banner {cls}'><div class='risk-title'>{title}</div>{body}</div>",
        unsafe_allow_html=True,
    )

def persona_card(name, action):
    icon = PERSONA_ICONS.get(name, "👤")
    st.markdown(
        f"<div class='persona-card'><div class='persona-name'>{icon} {name}</div>"
        f"<div class='persona-action'><b>Recommended action:</b> {action}</div></div>",
        unsafe_allow_html=True,
    )

def pretty_bar(data, y_title="", colors=None, sign_colors=False, horizontal=False):
    """Rounded, coloured bar chart. data = Series or one-column DataFrame."""
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    d = pd.DataFrame({"label": [str(i) for i in data.index], "value": data.values.astype(float)})
    order = list(d["label"])
    if sign_colors:
        d["color"] = np.where(d["value"] > 0, "#E17055", "#00B894")
    elif colors:
        d["color"] = [colors[i % len(colors)] for i in range(len(d))]
    else:
        d["color"] = "#6C5CE7"
    tip = [alt.Tooltip("label:N", title=""), alt.Tooltip("value:Q", format=",.1f", title=y_title or "Value")]
    if horizontal:
        chart = (
            alt.Chart(d)
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
                x=alt.X("value:Q", title=y_title),
                color=alt.Color("color:N", scale=None, legend=None),
                tooltip=tip,
            )
            .properties(height=max(160, 34 * len(d)))
        )
    else:
        chart = (
            alt.Chart(d)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
                y=alt.Y("value:Q", title=y_title),
                color=alt.Color("color:N", scale=None, legend=None),
                tooltip=tip,
            )
            .properties(height=320)
        )
    st.altair_chart(chart, use_container_width=True)

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

@st.cache_data
def get_test_results():
    """True outcomes and predicted risks for the 600 held-out test learners
    (same 80/20 stratified split, random_state=42, as in the training notebook)."""
    X_all = pd.get_dummies(
        df.drop(columns=['churned', 'days_to_churn']),
        columns=['occupation_type', 'signup_goal', 'plan_type', 'signup_discount_used'],
    )
    X_all = X_all.reindex(columns=model_columns, fill_value=0).astype(float)
    y_all = df['churned'].values
    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    probs = model.predict_proba(scaler.transform(X_test))[:, 1]
    return y_test, probs

def metrics_at(y_true, probs, t):
    """Confusion matrix counts and scores when flagging everyone with risk >= t."""
    pred = (probs >= t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return tp, fp, fn, tn, acc, prec, rec, f1

# ---- Survival analysis helpers (Kaplan-Meier, written in plain pandas/numpy) ----
OBSERVATION_DAYS = 365  # learners who never churned are followed for up to 365 days, then "censored"

def kaplan_meier(durations, events, max_day=OBSERVATION_DAYS):
    """Kaplan-Meier survival curve: for each day 0..max_day, the share of learners still subscribed.
    durations = days until churn (or until the end of follow-up if they never churned)
    events    = 1 if the learner churned, 0 if they were still subscribed at the end (censored)"""
    d = pd.DataFrame({"t": np.asarray(durations, dtype=float), "e": np.asarray(events, dtype=int)})
    n = len(d)
    if n == 0:
        return np.ones(max_day + 1)
    all_at_t = d.groupby("t").size().sort_index()
    churn_at_t = d[d["e"] == 1].groupby("t").size().reindex(all_at_t.index, fill_value=0)
    at_risk = n - all_at_t.cumsum() + all_at_t          # learners still under observation at each time
    surv = (1 - churn_at_t / at_risk).cumprod()          # chance of surviving each churn time, multiplied
    days = np.arange(0, max_day + 1)
    pos = np.searchsorted(surv.index.values, days, side="right") - 1
    return np.where(pos < 0, 1.0, surv.values[np.maximum(pos, 0)])

@st.cache_data
def survival_curves(group_col):
    """One curve per group (or a single curve for everyone). Returns a long table."""
    work = df.copy()
    work["duration"] = work["days_to_churn"].fillna(OBSERVATION_DAYS)
    days = np.arange(0, OBSERVATION_DAYS + 1)
    rows = []
    if group_col is None:
        groups = [("All learners", work)]
    else:
        groups = [(str(name), g) for name, g in work.groupby(group_col)]
    for name, g in groups:
        s = kaplan_meier(g["duration"], g["churned"])
        rows.append(pd.DataFrame({
            "Group": name, "Day": days, "Still subscribed (%)": s * 100, "Learners": len(g),
        }))
    return pd.concat(rows, ignore_index=True)

hero("📚 EdTech Learner Churn Dashboard",
     "Predict who will leave, understand why, and know what to do about it, before they go.")

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

    f1c, f2c, f3c = st.columns(3)
    feature_cards = [
        ("🔮", "Predict", "See who is likely to leave, with a 🟢🟡🔴 risk level."),
        ("🧠", "Understand", "Find out why, with plain-English reasons and learner personas."),
        ("💰", "Act", "Test what-if changes and see whether a retention campaign pays off."),
    ]
    for col, (icon, title, text) in zip((f1c, f2c, f3c), feature_cards):
        col.markdown(
            f"<div class='feature-card'><div class='fc-icon'>{icon}</div>"
            f"<div class='fc-title'>{title}</div><div class='fc-text'>{text}</div></div>",
            unsafe_allow_html=True,
        )

    st.subheader("What each page does")
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
- **Overview:** the big numbers and the biggest patterns at a glance
- **Explore:** compare groups and behaviours, and see what separates learners who stay from those who leave
- **Predict:** describe any learner (or load an example) and get their risk (🟢 low, 🟡 medium, 🔴 high)
- **Why?:** what drives churn overall, and the reasons behind one learner's risk, plus their persona
            """
        )
    with b:
        st.markdown(
            """
- **Model Performance:** how good the model really is, and how the decision threshold trades missed leavers against false alarms
- **Survival Curves:** *when* learners leave, and how long different groups stay subscribed
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

    with st.expander("Experiment 6: Catch more leavers, or raise fewer false alarms? (Model Performance)"):
        st.markdown(
            """
1. Open **Model Performance** and drag the **decision threshold** slider down to *0.30*, then up to *0.70*.
2. Watch the "leavers caught" and "false alarms" numbers move in opposite directions.
3. In the cost section, make a missed leaver much more expensive than a false alarm. What happens to the best threshold?

*What you learn:* there is no free lunch. The right setting depends on what each mistake costs the business.
            """
        )

    with st.expander("Experiment 7: When do learners leave? (Survival Curves)"):
        st.markdown(
            """
1. Open **Survival Curves** and compare **Plan type**. Which curve drops faster in the first 100 days?
2. Move the **day** slider to *30*, then *90*, then *180*. How many learners are still subscribed in each group?
3. Switch to **Signup goal** or **Signup discount**. Do those groups separate as clearly as the plans do?

*What you learn:* churn is not only about *who* leaves but *when*. Early weeks are the danger zone.
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
        "Precision": "Of all learners the model flagged, how many really would have left. Low precision means many false alarms.",
        "Decision threshold": "The risk level above which a learner is flagged. A lower threshold flags more learners: more leavers caught, but more false alarms.",
        "Survival curve": "A line showing the share of learners still subscribed after each number of days. It only goes down, and a steeper drop means people leave faster.",
        "Censoring": "Learners who had not left by the end of the study. We do not know when (or if) they will leave, so they count as 'still subscribed' instead of being thrown away.",
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
    pretty_bar(plan_churn, "Churn rate (%)", colors=["#6C5CE7", "#A29BFE"])
    st.success(
        f"Takeaway: {plan_churn.idxmax()} learners churn the most "
        f"({plan_churn.max():.1f}%), compared with {plan_churn.min():.1f}% for {plan_churn.idxmin()}."
    )

    st.subheader("Churn rate by engagement trend (%)")
    trend_labels = {-1: 'Declining', 0: 'Flat', 1: 'Improving'}
    trend_churn = df.groupby('engagement_trend')['churned'].mean() * 100
    trend_churn.index = trend_churn.index.map(trend_labels)
    pretty_bar(trend_churn, "Churn rate (%)", colors=["#E17055", "#B2BEC3", "#00B894"])
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
    pretty_bar(rates, "Churn rate (%)", colors=PALETTE)
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
    pretty_bar(compare, "Average", colors=["#00B894", "#E17055"])
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
    risk_banner(risk)

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
        {"Effect": top.values},
        index=[SHORT_LABELS[c] for c in top.index],
    )
    pretty_bar(chart, "Effect on churn risk (red = raises risk, green = lowers risk)",
               sign_colors=True, horizontal=True)

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

    risk_banner(risk, long=False)

    # Reasons for this learner (linear model: contribution = coefficient x scaled value)
    contrib = pd.Series(model.coef_[0] * input_scaled[0], index=model_columns)
    contrib = contrib[[c for c in contrib.index if c in SHORT_LABELS]]
    top_contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
    contrib_chart = pd.DataFrame(
        {"Push": top_contrib.values},
        index=[SHORT_LABELS[c] for c in top_contrib.index],
    )
    pretty_bar(contrib_chart, "Push on risk (red = toward leaving, green = toward staying)",
               sign_colors=True, horizontal=True)

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
        st.subheader("Persona")
        persona_card(info["name"], info["action"])
    else:
        st.caption("Personas only apply to learners at medium or high risk.")

    st.caption("Note: the model is tuned to catch as many churners as possible, so its percentages run higher than real-world churn rates. Use them to rank learners, not as exact probabilities.")

# ======================= MODEL PERFORMANCE =======================
elif page == "Model Performance":
    st.header("Model performance: how good is the model, really?")
    st.info(
        "How to read this: the model is graded only on 600 learners it never saw during training. "
        "Drag the threshold slider to decide how cautious the model should be."
    )

    y_test, probs_test = get_test_results()

    # ---------------- Part 1: report card ----------------
    st.subheader("1. The report card")
    threshold = st.slider(
        "Decision threshold: flag a learner as 'likely to leave' when their risk is at least",
        0.05, 0.95, 0.50, 0.05,
        help="Lower = flag more learners (catch more leavers, more false alarms). Higher = flag fewer.",
    )

    tp, fp, fn, tn, acc, prec, rec, f1 = metrics_at(y_test, probs_test, threshold)
    auc = roc_auc_score(y_test, probs_test)

    st.markdown("**Learners who actually left**")
    left1, left2 = st.columns(2)
    left1.metric("✅ Leavers the model caught", f"{tp}")
    left2.metric("❌ Leavers the model missed", f"{fn}")

    st.markdown("**Learners who actually stayed**")
    stay1, stay2 = st.columns(2)
    stay1.metric("⚠️ Stayers flagged by mistake (false alarms)", f"{fp}")
    stay2.metric("✅ Stayers correctly left alone", f"{tn}")

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Accuracy", f"{acc * 100:.0f}%")
    s2.metric("Precision", f"{prec * 100:.0f}%")
    s3.metric("Recall", f"{rec * 100:.0f}%")
    s4.metric("F1 score", f"{f1 * 100:.0f}%")
    s5.metric("AUC", f"{auc:.2f}")

    st.success(
        f"Takeaway: at a threshold of {threshold:.2f}, the model catches {rec * 100:.0f}% of the learners who leave, "
        f"but {100 - prec * 100:.0f}% of the learners it flags would have stayed anyway."
    )

    with st.expander("What do these numbers mean?"):
        st.markdown(
            """
- **Accuracy:** how often the model is right overall. It can look good even when the model misses most leavers, because most learners stay.
- **Precision:** of the learners the model flags, how many really leave. Low precision means many wasted retention offers.
- **Recall:** of the learners who really leave, how many the model catches. Low recall means silent revenue loss.
- **F1 score:** one number that balances precision and recall.
- **AUC:** pick one leaver and one stayer at random. AUC is how often the model gives the leaver the higher risk. 0.50 is a coin flip, 1.00 is perfect.
            """
        )

    st.caption(
        "Check: at threshold 0.50 you should see 104 leavers caught, 34 missed, 104 false alarms and 358 stayers correctly "
        "left alone, the same as the training notebook (a difference of one or two learners is just rounding)."
    )

    # ---------------- Part 2: trade-off chart ----------------
    st.divider()
    st.subheader("2. The trade-off: catching leavers vs false alarms")

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sweep_rows = []
    for t in grid:
        s_tp, s_fp, s_fn, s_tn, s_acc, s_prec, s_rec, s_f1 = metrics_at(y_test, probs_test, t)
        sweep_rows.append({
            "Threshold": float(t), "Precision": s_prec * 100, "Recall": s_rec * 100, "F1": s_f1 * 100,
            "FN": s_fn, "FP": s_fp,
        })
    sweep = pd.DataFrame(sweep_rows)

    sweep_long = sweep.melt(id_vars="Threshold", value_vars=["Precision", "Recall", "F1"],
                            var_name="Metric", value_name="Percent")
    lines = (
        alt.Chart(sweep_long)
        .mark_line(point=True)
        .encode(
            x=alt.X("Threshold:Q", title="Decision threshold"),
            y=alt.Y("Percent:Q", title="Percent"),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(domain=["Precision", "Recall", "F1"], range=["#6C5CE7", "#E17055", "#00B894"]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[alt.Tooltip("Threshold:Q", format=".2f"), alt.Tooltip("Metric:N"),
                     alt.Tooltip("Percent:Q", format=".1f")],
        )
    )
    marker = (
        alt.Chart(pd.DataFrame({"Threshold": [float(threshold)]}))
        .mark_rule(color="#2D1B69", strokeDash=[5, 4], size=2)
        .encode(x="Threshold:Q")
    )
    st.altair_chart((lines + marker).properties(height=320), use_container_width=True)
    st.success(
        "Takeaway: as the threshold goes down, recall rises (fewer leavers slip through) but precision falls "
        "(more false alarms). The dashed line shows your chosen threshold."
    )

    # ---------------- Part 3: cost of mistakes ----------------
    st.divider()
    st.subheader("3. What do the mistakes cost?")
    st.info("How to read this: set what each kind of mistake costs, and find the threshold with the lowest total cost.")

    cc1, cc2 = st.columns(2)
    miss_cost = cc1.number_input(
        f"Cost of missing a learner who leaves ({SYM}, about 6 months of fees)",
        min_value=1, max_value=1000000, value=DEF_MONTHLY * 6, key=f"miss_{SYM}",
    )
    alarm_cost = cc2.number_input(
        f"Cost of a false alarm: one wasted retention offer ({SYM})",
        min_value=1, max_value=1000000, value=DEF_OFFER, key=f"alarm_{SYM}",
    )

    sweep["Cost"] = sweep["FN"] * miss_cost + sweep["FP"] * alarm_cost
    best_row = sweep.loc[sweep["Cost"].idxmin()]
    best_t = float(best_row["Threshold"])
    best_cost = float(best_row["Cost"])
    sel_cost = float(fn * miss_cost + fp * alarm_cost)

    q1, q2, q3 = st.columns(3)
    q1.metric(f"Total cost at {threshold:.2f}", fmt_money(sel_cost))
    q2.metric("Lowest possible total cost", fmt_money(best_cost))
    q3.metric("Cheapest threshold", f"{best_t:.2f}")

    cost_line = (
        alt.Chart(sweep)
        .mark_line(point=True, color="#6C5CE7")
        .encode(
            x=alt.X("Threshold:Q", title="Decision threshold"),
            y=alt.Y("Cost:Q", title=f"Total cost of mistakes ({SYM})"),
            tooltip=[alt.Tooltip("Threshold:Q", format=".2f"), alt.Tooltip("Cost:Q", format=",.0f")],
        )
    )
    cost_marker = (
        alt.Chart(pd.DataFrame({"Threshold": [float(threshold)]}))
        .mark_rule(color="#2D1B69", strokeDash=[5, 4], size=2)
        .encode(x="Threshold:Q")
    )
    st.altair_chart((cost_line + cost_marker).properties(height=300), use_container_width=True)

    ratio = miss_cost / alarm_cost
    if abs(best_t - threshold) < 1e-9:
        st.success(
            f"Takeaway: a missed leaver costs {ratio:.0f}x a false alarm here, and your chosen threshold of "
            f"{threshold:.2f} is already the cheapest one."
        )
    else:
        st.success(
            f"Takeaway: a missed leaver costs {ratio:.0f}x a false alarm here, so the cheapest threshold is {best_t:.2f}, "
            f"with a total cost of {fmt_money(best_cost)} on these 600 learners (versus {fmt_money(sel_cost)} at {threshold:.2f})."
        )
    st.caption(
        "Simplified: every missed leaver is assumed to be lost, every false alarm wastes one offer, and the cost of offers "
        "sent to correctly flagged learners is ignored."
    )

    # ---------------- Part 4: model comparison ----------------
    st.divider()
    st.subheader("4. Why this model?")
    comparison = pd.DataFrame(
        {
            "Accuracy": ["83.5%", "77%", "82%", "81%"],
            "Leavers caught (recall)": ["54%", "75%", "67%", "64%"],
            "Precision": ["68%", "50%", "60%", "58%"],
            "Leavers missed": [63, 34, 46, 49],
        },
        index=[
            "Logistic Regression",
            "Logistic Regression, balanced (used here)",
            "Random Forest",
            "XGBoost",
        ],
    )
    st.dataframe(comparison, use_container_width=True)
    st.success(
        "Takeaway: the simplest model with class weighting caught the most leavers. The more complex models did not win, "
        "because the synthetic data follows clean, additive patterns."
    )
    st.caption(
        "Comparison numbers come from the training notebook, on the same 600 test learners. "
        "All results are on synthetic data, so real-world performance would differ."
    )

# ========================= SURVIVAL CURVES =========================
elif page == "Survival Curves":
    st.header("Survival curves: when do learners leave?")
    st.info(
        "How to read this: each line shows the share of learners still subscribed after a number of days. "
        "A line that drops faster means that group leaves sooner."
    )

    survival_groups = {
        "All learners": None,
        "Plan type": "plan_type",
        "Signup goal": "signup_goal",
        "Occupation": "occupation_type",
        "Used a signup discount": "signup_discount_used",
    }
    group_choice = st.selectbox("Compare curves by:", list(survival_groups.keys()), index=1)
    curves = survival_curves(survival_groups[group_choice])
    group_names = list(curves["Group"].drop_duplicates())

    # ---------------- Part 1: the curves ----------------
    st.subheader("1. Retention curves")
    color_range = (["#6C5CE7"] if len(group_names) == 1
                   else ["#E17055", "#6C5CE7", "#00B894", "#74B9FF"][:len(group_names)])
    day_pick = st.slider("Check the share still subscribed after this many days", 0, OBSERVATION_DAYS, 90, 10)

    curve_lines = (
        alt.Chart(curves)
        .mark_line(interpolate="step-after", strokeWidth=3)
        .encode(
            x=alt.X("Day:Q", title="Days since signup", scale=alt.Scale(domain=[0, OBSERVATION_DAYS])),
            y=alt.Y("Still subscribed (%):Q", title="Still subscribed (%)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "Group:N",
                scale=alt.Scale(domain=group_names, range=color_range),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[alt.Tooltip("Group:N"), alt.Tooltip("Day:Q"),
                     alt.Tooltip("Still subscribed (%):Q", format=".1f")],
        )
    )
    day_marker = (
        alt.Chart(pd.DataFrame({"Day": [int(day_pick)]}))
        .mark_rule(color="#2D1B69", strokeDash=[5, 4], size=2)
        .encode(x="Day:Q")
    )
    st.altair_chart((curve_lines + day_marker).properties(height=340), use_container_width=True)

    # ---------------- Part 2: readout at the chosen day ----------------
    st.subheader(f"2. Who is still subscribed after {day_pick} days?")
    at_day = curves[curves["Day"] == day_pick].set_index("Group")
    cols = st.columns(len(group_names))
    for col, name in zip(cols, group_names):
        col.metric(name, f"{at_day.loc[name, 'Still subscribed (%)']:.1f}%",
                   help=f"{int(at_day.loc[name, 'Learners']):,} learners in this group")

    if len(group_names) > 1:
        best = at_day["Still subscribed (%)"].idxmax()
        worst = at_day["Still subscribed (%)"].idxmin()
        gap = at_day.loc[best, "Still subscribed (%)"] - at_day.loc[worst, "Still subscribed (%)"]
        st.success(
            f"Takeaway: after {day_pick} days, '{best}' learners are the most likely to still be subscribed "
            f"({at_day.loc[best, 'Still subscribed (%)']:.1f}%), and '{worst}' the least "
            f"({at_day.loc[worst, 'Still subscribed (%)']:.1f}%), a gap of {gap:.1f} points."
        )
    else:
        st.success(
            f"Takeaway: after {day_pick} days, about {at_day['Still subscribed (%)'].iloc[0]:.1f}% of all learners "
            f"are still subscribed."
        )

    # ---------------- Part 3: when does the leaving happen? ----------------
    st.divider()
    st.subheader("3. When does most churn happen?")
    churners = df[df["churned"] == 1]["days_to_churn"]
    bands = pd.cut(
        churners,
        bins=[0, 30, 60, 100, 180, OBSERVATION_DAYS],
        labels=["Days 1-30", "Days 31-60", "Days 61-100", "Days 101-180", "Days 181-365"],
    )
    band_share = (bands.value_counts(normalize=True).reindex(bands.cat.categories) * 100)
    pretty_bar(band_share, "Share of all churners (%)", colors=["#E17055", "#6C5CE7"])
    first_100 = band_share.iloc[:3].sum()
    st.success(
        f"Takeaway: {first_100:.0f}% of the learners who churn do so within their first 100 days. "
        f"That is the window where retention effort should go first."
    )

    with st.expander("How is this calculated? (Kaplan-Meier, in plain English)"):
        st.markdown(
            f"""
- Every learner has a **duration**: the days until they churned. Learners who never churned have no churn day, so they are
  followed for **{OBSERVATION_DAYS} days** and then marked as **censored** (still subscribed when we stopped watching).
- On each day that someone churns, the curve multiplies by *(1 minus churners that day divided by learners still in the study)*.
- Censored learners stay in the calculation until the end, so we do not pretend they left and we do not throw them away.
- This is the **Kaplan-Meier estimator**, the standard tool for "how long until an event happens" questions. It is written here in plain pandas and numpy.
            """
        )

    st.caption(
        "Assumption: learners who did not churn are treated as observed for 365 days. The data is synthetic, so the shape of "
        "the curves shows how the method works, not how a real company behaves. Curves describe groups, not single learners."
    )

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
        {"Amount": [do_nothing_loss, loss_with_campaign, campaign_cost]},
        index=["Revenue lost: do nothing", "Revenue lost: run campaign", "Campaign cost"],
    )
    pretty_bar(compare_chart, f"Amount ({SYM})", colors=["#E17055", "#FDCB6E", "#6C5CE7"])

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
        "Days since last login": clean["days_since_last_login"].astype(int).values,
        "Course completion (%)": clean["course_completion_rate"].round(0).astype(int).values,
        "Satisfaction (1-5)": clean["satisfaction_score"].astype(int).values,
        "Engagement trend": clean["engagement_trend"].map({-1: "Declining", 0: "Flat", 1: "Improving"}).values,
        "Plan": clean["plan_type"].values,
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

    pretty_bar(counts, "Learners", colors=["#00B894", "#FDCB6E", "#E17055"])

    n_act = int(counts["🔴 High"] + counts["🟡 Medium"])
    st.success(
        f"Takeaway: {n_act} of {len(results)} learners ({n_act / len(results) * 100:.0f}%) need attention. "
        f"The table below is sorted so the learners to contact first are at the top."
    )

    def color_risk(col):
        return [
            "background-color: #FFE6E0" if "High" in v
            else "background-color: #FFF6DB" if "Medium" in v
            else "background-color: #E3F9F1"
            for v in col
        ]

    st.dataframe(
        results.style.apply(color_risk, subset=["Risk level"]).format({"Churn risk (%)": "{:.1f}"}),
        use_container_width=True,
    )

    # Excel-friendly download: plain words instead of emoji, and a marker so Excel reads it correctly
    csv_df = results.copy()
    csv_df["Risk level"] = csv_df["Risk level"].str.replace(r"[^\x00-\x7F]+", "", regex=True).str.strip()
    st.download_button(
        "⬇ Download results as CSV",
        csv_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="churn_risk_results.csv",
        mime="text/csv",
    )

    st.caption(
        "Privacy: this app does not save uploaded files. The model was trained on synthetic data, so results show how "
        "the method works and should not be used for real decisions. Please do not upload real personal data."
    )

# ============================ FOOTER ============================
st.markdown(
    "<div class='app-footer'>Built as a data science portfolio project · Synthetic data, for learning and demonstration</div>",
    unsafe_allow_html=True,
)
