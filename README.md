# 📚 EdTech Learner Churn Prediction & Retention Dashboard

A full end-to-end churn analytics pipeline for an EdTech platform — from raw learner
data to a live, interactive business decision tool.

**Live app:** https://churn-prediction-dashboard-uh4q8f9sqxb8frx4wi7kjg.streamlit.app

## The Problem
The platform was losing an estimated **$20,576** in expected revenue from at-risk
learners with no way to know who, why, or what to do about it.

## The Solution
1. **Predict** — Logistic Regression model flags learners likely to churn (77% accuracy, tuned for recall on the churn class)
2. **Explain** — Model coefficients show which behaviors drive risk up or down (e.g. days since last login, course completion rate)
3. **Segment** — K-Means clustering groups at-risk learners into 4 actionable personas (Passive Coaster, Goal-Focused Drifter, Quiet Achiever, Struggling but Trying), each with a tailored recommended action
4. **Value** — Customer Lifetime Value (CLV) estimates translate "risk" into dollars
5. **Act** — An ROI simulator shows that targeting the top 20% highest-priority at-risk learners costs **$472** and protects **$5,173** in revenue — a **996% ROI**

## Dashboard Pages
- **Overview** — key stats and persona breakdown
- **Explore** — churn patterns by plan type, engagement trend, completion rate
- **Predict** — enter any learner's details, get a live risk score + persona
- **Why?** — see which factors drive churn risk up or down, and explain any individual learner
- **Retention Simulator** — adjust campaign cost, success rate, and targeting % to see projected ROI

## Tech Stack
Python · pandas · scikit-learn (Logistic Regression, KMeans) · lifelines (survival analysis) · Streamlit

## Data
Synthetic dataset of 3,000 learners with behavioral, engagement, and outcome features
(quiz scores, login recency, completion rate, support tickets, etc.), churn labels,
and time-to-churn.
