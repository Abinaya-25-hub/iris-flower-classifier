import streamlit as st
import numpy as np
import joblib

# === Page Title ===
st.title("🌸 Iris Flower Classifier")

# === Load the trained model ===
model = joblib.load("iris_model.pkl")

# === Input sliders ===
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# === Prepare input ===
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# === Predict button ===
if st.button("Predict"):
    prediction = model.predict(features)[0]
    classes = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Predicted Iris Species: {classes[prediction]}")
