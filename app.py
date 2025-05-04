import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Load model tflite
interpreter = tf.lite.Interpreter(model_path="iris_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit UI
st.title("Prediksi Jenis Bunga Iris 🌸")

st.write("Masukkan ukuran sepal dan petal bunga:")

# Input
sepal_length = st.number_input("Sepal Length", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, step=0.1)

if st.button("Prediksi"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)

    # Masukkan input ke model tflite
    interpreter.set_tensor(input_details[0]['index'], input_scaled.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred_label = np.argmax(output, axis=1)
    species = le.inverse_transform(pred_label)[0]

    st.success(f"Jenis bunga yang diprediksi adalah: **{species}**")
