import streamlit as st
import pandas as pd
import joblib

model = joblib.load("iris_modell.pkl")

st.title("Iris Flower Species Predictor")
st.write("Adjust the sliders on the left to predict the Iris species.")

#sidebar
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length(CM)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width(CM)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length(CM)", 1.0, 7.0, 4.3)
petal_width = st.sidebar.slider("Petal Width(CM)", 0.1, 2.5, 1.3)

input_data = pd.DataFrame({
    "SepalLengthCm": [sepal_length],
    "SepalWidthCm": [sepal_width],
    "PetalLengthCm": [petal_length],
    "PetalWidthCm": [petal_width]
})

st.write(" User Input Features")
st.dataframe(input_data)


prediction = model.predict(input_data)
predicted_species = prediction[0]

st.write("## Prediction")
st.subheader({predicted_species})