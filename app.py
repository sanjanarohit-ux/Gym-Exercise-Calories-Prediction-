import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Gym Exercise Tracker",
    page_icon="🏋️",
    layout="wide"
)

# =========================================
# LOAD DATA & MODEL
# =========================================
@st.cache_data
def load_data():
    return pd.read_csv("gym_members_exercise_tracking.csv")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

df = load_data()
model = load_model()

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.title {
    font-size: 45px;
    font-weight: bold;
    color: #00F5D4;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #A0AEC0;
    margin-bottom: 30px;
}

.metric-card {
    background: linear-gradient(135deg, #1F2937, #111827);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
}

.stButton>button {
    background: linear-gradient(90deg, #00F5D4, #00BBF9);
    color: black;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
    padding: 12px 25px;
    border: none;
    width: 100%;
}

.stButton>button:hover {
    color: white;
    transform: scale(1.02);
    transition: 0.3s;
}

.prediction-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================
st.markdown(
    '<div class="title">🏋️ Gym Exercise Tracking Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Powered Fitness Prediction System</div>',
    unsafe_allow_html=True
)

# =========================================
# SIDEBAR
# =========================================
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2936/2936886.png",
    width=120
)

st.sidebar.header("⚡ User Input")

# =========================================
# INPUT FIELDS
# =========================================
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 15, 70, 25)
    weight = st.slider("Weight (kg)", 40, 150, 70)
    height = st.slider("Height (m)", 1.3, 2.2, 1.75)
    max_bpm = st.slider("Max BPM", 60, 220, 180)

with col2:
    avg_bpm = st.slider("Average BPM", 50, 200, 120)
    resting_bpm = st.slider("Resting BPM", 40, 100, 70)
    session_duration = st.slider(
        "Session Duration (hrs)",
        0.5,
        5.0,
        1.5
    )
    calories_burned = st.slider(
        "Calories Burned",
        100,
        2000,
        500
    )

fat_percentage = st.slider(
    "Fat Percentage",
    5.0,
    40.0,
    18.0
)

water_intake = st.slider(
    "Water Intake (liters)",
    1.0,
    10.0,
    3.0
)

workout_frequency = st.slider(
    "Workout Frequency/week",
    1,
    14,
    5
)

experience_level = st.selectbox(
    "Experience Level",
    [1, 2, 3]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

# =========================================
# ENCODE GENDER
# =========================================
gender_male = 1 if gender == "Male" else 0

# =========================================
# EXTRA FEATURES FOR 17 INPUTS
# =========================================
bmi = weight / (height ** 2)

heart_rate_reserve = max_bpm - resting_bpm

intensity = avg_bpm / max_bpm

hydration_ratio = water_intake / weight

calories_per_hour = calories_burned / session_duration

# =========================================
# CREATE INPUT ARRAY (17 FEATURES)
# =========================================
input_data = np.array([[
    age,
    weight,
    height,
    max_bpm,
    avg_bpm,
    resting_bpm,
    session_duration,
    calories_burned,
    fat_percentage,
    water_intake,
    workout_frequency,
    experience_level,
    gender_male,
    bmi,
    heart_rate_reserve,
    intensity,
    calories_per_hour
]])

# =========================================
# PREDICTION
# =========================================
if st.button("🚀 Predict Fitness Score"):

    prediction = model.predict(input_data)[0]

    st.markdown(
        f"""
        <div class="prediction-box" style="
        background: linear-gradient(135deg,#00F5D4,#00BBF9);
        color:black;">
        Predicted Result: {round(prediction,2)}
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================
# DATA OVERVIEW
# =========================================
st.markdown("## 📊 Dataset Overview")

col3, col4, col5 = st.columns(3)

with col3:
    st.metric("Total Records", len(df))

with col4:
    st.metric("Total Features", df.shape[1])

with col5:
    st.metric(
        "Average Calories",
        int(df["Calories_Burned"].mean())
    )

# =========================================
# SHOW DATA
# =========================================
st.markdown("## 📁 Sample Dataset")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# =========================================
# FOOTER
# =========================================
st.markdown("---")

st.markdown(
    "<center>Made with ❤️ using Streamlit & Machine Learning</center>",
    unsafe_allow_html=True
)