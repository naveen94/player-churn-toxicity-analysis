import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Player Analytics", page_icon="🎮", layout="wide")

players = pd.read_csv("data/player_data.csv")
chats = pd.read_csv("data/chat_data.csv")

st.title("Player Churn & Toxicity Analytics")
st.caption("Data & Business Analytics project — retention intelligence and community safety")

churn_rate = players["churned"].mean() * 100
toxicity_rate = chats["toxic"].mean() * 100
avg_sessions = players["sessions_per_week"].mean()
avg_inactive = players["days_inactive"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Players", f"{len(players):,}")
c2.metric("Churn Rate", f"{churn_rate:.1f}%")
c3.metric("Toxic Messages", f"{toxicity_rate:.1f}%")
c4.metric("Avg Inactive Days", f"{avg_inactive:.1f}")

tab1, tab2, tab3 = st.tabs(["Churn Analysis", "Toxicity Detection", "Business Insights"])

with tab1:
    st.subheader("Churn Drivers")
    col1, col2 = st.columns(2)

    with col1:
        st.write("Churn by inactivity")
        chart = players.groupby("days_inactive")["churned"].mean().reset_index()
        st.line_chart(chart.set_index("days_inactive"))

    with col2:
        st.write("Churn by sessions per week")
        chart2 = players.groupby("sessions_per_week")["churned"].mean().reset_index()
        st.line_chart(chart2.set_index("sessions_per_week"))

    st.write("Player segments")
    st.dataframe(
        players.groupby("churned")[[
            "sessions_per_week", "avg_session_minutes",
            "days_inactive", "win_rate", "purchases",
            "toxicity_rate", "friends_count"
        ]].mean().round(2),
        use_container_width=True
    )

    st.subheader("Predict Churn Risk")
    try:
        bundle = joblib.load("models/churn_model.pkl")
        model = bundle["model"]
        features = bundle["features"]

        left, right = st.columns(2)
        with left:
            age = st.slider("Age", 13, 60, 22)
            days_signup = st.slider("Days Since Signup", 30, 1000, 250)
            sessions = st.slider("Sessions / Week", 0.5, 15.0, 5.0)
            minutes = st.slider("Average Session Minutes", 5, 180, 45)
            inactive = st.slider("Days Inactive", 0, 60, 5)
            matches = st.slider("Matches Played", 1, 150, 25)
            win = st.slider("Win Rate", 0.05, 0.95, 0.50)

        with right:
            purchases = st.slider("Purchases", 0, 30, 2)
            avg_value = st.slider("Average Purchase Value", 0.0, 100.0, 20.0)
            tickets = st.slider("Support Tickets", 0, 15, 1)
            toxicity = st.slider("Toxicity Rate", 0.0, 1.0, 0.05)
            friends = st.slider("Friends Count", 0, 30, 8)

        weekly_hours = sessions * minutes / 60
        row = pd.DataFrame([[
            age, days_signup, sessions, minutes, weekly_hours, inactive,
            matches, win, purchases, avg_value, tickets, toxicity, friends
        ]], columns=features)

        risk = model.predict_proba(row)[0,1]
        st.metric("Estimated Churn Probability", f"{risk*100:.1f}%")
        if risk >= 0.60:
            st.warning("High-risk segment: consider a retention intervention.")
        elif risk >= 0.35:
            st.info("Medium-risk segment: monitor engagement.")
        else:
            st.success("Lower-risk segment based on the model.")

    except FileNotFoundError:
        st.info("Run the model training commands in the README first.")

with tab2:
    st.subheader("Toxicity Overview")
    st.bar_chart(chats["toxic"].value_counts().rename({0:"Non-toxic", 1:"Toxic"}))
    st.write("Try a message:")
    message = st.text_area("Chat message", "good game everyone")
    try:
        tox_model = joblib.load("models/toxicity_model.pkl")
        pred = int(tox_model.predict([message])[0])
        prob = float(tox_model.predict_proba([message])[0, pred])
        if pred:
            st.error(f"Toxic message detected — confidence {prob*100:.1f}%")
        else:
            st.success(f"Non-toxic message — confidence {prob*100:.1f}%")
    except FileNotFoundError:
        st.info("Run the toxicity model training command in the README first.")

with tab3:
    st.subheader("Business Insights")
    st.markdown(f'''
    **Retention:** The dataset contains {len(players):,} players, with an observed churn rate of
    **{churn_rate:.1f}%**. Inactivity, engagement frequency, and social connection are useful
    variables for retention analysis.

    **Community health:** About **{toxicity_rate:.1f}%** of sample chat messages are labeled toxic.
    Toxicity can be monitored as a community-health KPI alongside player retention.

    **Recommended analytics workflow:** identify high-risk players, segment them by engagement,
    investigate likely churn drivers, and evaluate retention campaigns using controlled experiments.
    ''')

    st.subheader("Top Business KPIs")
    kpis = pd.DataFrame({
        "KPI": ["Churn Rate", "Toxic Message Rate", "Average Sessions/Week", "Average Inactive Days"],
        "Value": [
            f"{churn_rate:.1f}%",
            f"{toxicity_rate:.1f}%",
            f"{avg_sessions:.2f}",
            f"{avg_inactive:.1f}"
        ]
    })
    st.table(kpis)
