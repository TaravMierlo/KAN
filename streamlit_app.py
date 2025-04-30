import streamlit as st
import torch
import dill
import numpy as np

# Load the MultKAN model saved with dill
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        model = dill.load(f)
    model.eval()
    return model

# UI to upload model
model_path = st.text_input("Enter path to saved MultKAN model (.pkl):", "trained_multkan.pkl")

# Load model when button is clicked
if st.button("Load Model"):
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Collect input features from user
    st.subheader("Input Features")
    # Example: adjust to match your dataset's number of input features
    num_features = model.input_dim if hasattr(model, "input_dim") else 10
    input_data = []
    for i in range(num_features):
        val = st.number_input(f"Feature {i + 1}", value=0.0)
        input_data.append(val)

    if st.button("Predict"):
        x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(x)
            prediction = output.argmax(dim=1).item()
        st.write(f"🧠 Prediction: **{prediction}**")