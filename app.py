import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Set Parameters", "Calorie Burn Prediction", "Workout Plan Recommendation", "General Information"])

# Home Page
if page == "Home":
    st.title("🏋️ Personal Fitness Tracker")
    st.write("""
        Welcome to your **Personal Fitness Tracker**! 🚀
        
        This tool helps you:
        - ✅ Predict your calorie burn.
        - ✅ Get workout recommendations.
        - ✅ Track your exercise statistics compared to other users.

        Use the sidebar to navigate between different sections.
    """)

# User Input Page (Separate from Sidebar)
if page == "Set Parameters":
    st.title("⚙️ Set Your Fitness Parameters")
    
    age = st.slider("Age", 10, 100, 30)
    bmi = st.slider("BMI", 15, 40, 20)
    duration = st.slider("Workout Duration (minutes)", 0, 60, 20)
    heart_rate = st.slider("Heart Rate", 60, 180, 85)
    body_temp = st.slider("Body Temperature (°C)", 36, 42, 37)
    gender_button = st.radio("Gender", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0

    # Store input in session state
    st.session_state["user_params"] = {
        "Age": age, "BMI": bmi, "Duration": duration,
        "Heart_Rate": heart_rate, "Body_Temp": body_temp, "Gender": gender
    }
    st.success("✅ Parameters saved! Now, navigate to **Calorie Burn Prediction** or **Workout Plan**.")

# Ensure parameters are set before other pages
if "user_params" not in st.session_state:
    st.warning("⚠️ Please go to **Set Parameters** first.")
else:
    df = pd.DataFrame(st.session_state["user_params"], index=[0])

    # Simulate loading effect
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.005)

    # Load datasets
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")

    dataset = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])
    dataset["BMI"] = dataset["Weight"] / ((dataset["Height"] / 100) ** 2)

    dataset = dataset[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    dataset = pd.get_dummies(dataset, drop_first=True)

    X = dataset.drop("Calories", axis=1)
    y = dataset["Calories"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    df = df.reindex(columns=X_train.columns, fill_value=0)
    predicted_calories = model.predict(df)[0]

    # Page: Calorie Burn Prediction
    if page == "Calorie Burn Prediction":
        st.title("🔥 Predicted Calorie Burn")
        st.write("## Your estimated calorie burn:")
        st.write(f"### **{round(predicted_calories, 2)} kcal**")

    # Page: Workout Plan Recommendation
    elif page == "Workout Plan Recommendation":
        st.title("🏋️ Workout Plan Recommendation")
        if predicted_calories > 500:
            st.write("🏋️‍♂️ **High-intensity strength & endurance training**")
            st.write("Suggested: **Weight training + HIIT for 45-60 min**")
        elif predicted_calories > 300:
            st.write("🏃 **Moderate cardio & strength mix**")
            st.write("Suggested: **Jogging, cycling, and bodyweight exercises**")
        else:
            st.write("🚶 **Light movement & mobility workouts**")
            st.write("Suggested: **Walking, yoga, and stretching**")

        # Rest Day Suggestion
        st.write("## Should You Take a Rest Day?")
        if df["Heart_Rate"].values[0] > 140 or df["Body_Temp"].values[0] > 39:
            st.write("⚠️ Your heart rate or temperature is high. Consider taking a rest day!")
        elif df["Duration"].values[0] > 50:
            st.write("🔹 You've exercised a lot today. A rest day might help recovery.")
        else:
            st.write("✅ You're good to continue your workouts!")

    # Page: General Information
    elif page == "General Information":
        st.title("📊 General Information")

        boolean_age = (dataset["Age"] < df["Age"].values[0]).tolist()
        boolean_duration = (dataset["Duration"] < df["Duration"].values[0]).tolist()
        boolean_body_temp = (dataset["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
        boolean_heart_rate = (dataset["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

        st.write("📊 You are older than", round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other users.")
        st.write("⏳ Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other users.")
        st.write("❤️ Your heart rate is higher than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other users during exercise.")
        st.write("🌡️ Your body temperature is higher than", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other users during exercise.")
