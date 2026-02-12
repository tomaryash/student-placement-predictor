import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("placement_model.pkl")

st.title("🎓 Student Placement Prediction System")

cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
internships = st.number_input("Internships", 0, 10, 1)
projects = st.number_input("Projects", 0, 10, 2)
communication = st.number_input("Communication Score (1-10)", 1, 10, 5)
aptitude = st.number_input("Aptitude Score", 0, 100, 60)

if st.button("Predict Placement"):
    input_data = np.array([[cgpa, internships, projects, communication, aptitude]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Student is Likely to be Placed")
    else:
        st.error("❌ Student is Not Likely to be Placed")

# Placement Ratio
df = pd.read_csv("dataset.csv")
total = len(df)
placed = df["Placed"].sum()

st.write("### 📊 Placement Statistics")
st.write("Total Students:", total)
st.write("Placed Students:", placed)
st.write("Placement Ratio: {:.2f}%".format((placed/total)*100))
