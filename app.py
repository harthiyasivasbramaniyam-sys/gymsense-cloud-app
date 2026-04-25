import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, timedelta

st.set_page_config(page_title="GymSense Cloud App", page_icon="🏋️", layout="wide")

st.title("🏋️ GymSense: Cloud-Based Gym Attendance and Equipment Demand Prediction")
st.write(
    "This cloud application uses trained machine learning models to predict "
    "daily gym attendance and category-level equipment demand."
)

# Load models
attendance_model = joblib.load("attendance_model.pkl")
cardio_model = joblib.load("cardio_model.pkl")
strength_model = joblib.load("strength_model.pkl")
flexibility_model = joblib.load("flexibility_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")

# Load processed historical daily data
data = pd.read_csv("processed_daily_data.csv")
data["visit_date"] = pd.to_datetime(data["visit_date"])

st.sidebar.header("Prediction Input")

selected_date = st.sidebar.date_input(
    "Select prediction date",
    value=date.today() + timedelta(days=1),
    min_value=date.today() + timedelta(days=1)
)
is_holiday = st.sidebar.selectbox("Is this a holiday?", [0, 1])
is_promotion_day = st.sidebar.selectbox("Is this a promotion day?", [0, 1])

selected_date = pd.to_datetime(selected_date)

# Use latest row for lag/rolling values
latest = data.sort_values("visit_date").iloc[-1]

# Build input row
input_row = {}

input_row["day_of_week"] = selected_date.dayofweek
input_row["month"] = selected_date.month
input_row["week_of_year"] = int(selected_date.isocalendar().week)
input_row["day_of_month"] = selected_date.day
input_row["is_weekend"] = 1 if selected_date.dayofweek >= 5 else 0

input_row["is_holiday"] = is_holiday
input_row["is_promotion_day"] = is_promotion_day
input_row["is_special_day"] = 1 if (is_holiday == 1 or is_promotion_day == 1) else 0

input_row["sin_dow"] = np.sin(2 * np.pi * input_row["day_of_week"] / 7)
input_row["cos_dow"] = np.cos(2 * np.pi * input_row["day_of_week"] / 7)
input_row["sin_month"] = np.sin(2 * np.pi * input_row["month"] / 12)
input_row["cos_month"] = np.cos(2 * np.pi * input_row["month"] / 12)

# Continue trend from latest available row
input_row["trend"] = int(latest["trend"]) + 1 if "trend" in latest else len(data) + 1

# Fill lag and rolling features using latest known processed values
for col in feature_cols:
    if col not in input_row:
        if col in latest:
            input_row[col] = latest[col]
        else:
            input_row[col] = 0

X_input = pd.DataFrame([input_row])[feature_cols]

if st.button("Predict"):
    pred_attendance = attendance_model.predict(X_input)[0]
    pred_cardio = cardio_model.predict(X_input)[0]
    pred_strength = strength_model.predict(X_input)[0]
    pred_flexibility = flexibility_model.predict(X_input)[0]

    pred_attendance = max(0, round(pred_attendance))
    pred_cardio = max(0, round(pred_cardio))
    pred_strength = max(0, round(pred_strength))
    pred_flexibility = max(0, round(pred_flexibility))

    st.subheader("Prediction Results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Predicted Attendance", pred_attendance)
    col2.metric("Cardio Demand", pred_cardio)
    col3.metric("Strength Demand", pred_strength)
    col4.metric("Flexibility Demand", pred_flexibility)

    results_df = pd.DataFrame({
        "Prediction Output": [
            "Attendance",
            "Cardio Equipment Demand",
            "Strength Equipment Demand",
            "Flexibility Equipment Demand"
        ],
        "Predicted Value": [
            pred_attendance,
            pred_cardio,
            pred_strength,
            pred_flexibility
        ]
    })

    st.dataframe(results_df, use_container_width=True)

    st.info(
        "These predictions are generated using trained models from the GymSense pipeline. "
        "The application demonstrates cloud-based access to the forecasting framework."
    )

st.markdown("---")
st.caption("GymSense cloud deployment prototype using Streamlit.")