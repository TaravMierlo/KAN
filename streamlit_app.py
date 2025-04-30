import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle
from pykan.kan import KAN

# =========================
# Load preprocessing artifacts
# =========================
with open("models/scaler_cont.pkl", "rb") as f:
    scaler_cont = pickle.load(f)

with open("models/scaler_ord.pkl", "rb") as f:
    scaler_ord = pickle.load(f)

with open("models/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("models/feature_config.pkl", "rb") as f:
    feature_config = pickle.load(f)

continuous_labels = feature_config["continuous_labels"]
binary_labels = feature_config["binary_labels"]
ordinal_labels = feature_config["ordinal_labels"]

# =========================
# Load trained KAN model
# =========================
input_dim = len(continuous_labels) + len(binary_labels) + len(ordinal_labels)

# Use the exact same width you used during training
kan_model = KAN(width=[input_dim, 1, 2])  # adjust width if different

# Load model weights
kan_model.load_state_dict(torch.load("trained_kan_model.pt", map_location=torch.device("cpu")))
kan_model.eval()

# =========================
# Streamlit UI
# =========================
st.title("🧠 Predict SAD from MIMIC-IV")

st.markdown("Fill in the patient features below to predict Suicidal Attempt/Depression (SAD).")

# Collect user input
st.subheader("🧪 Continuous features")
user_cont = []
for label in continuous_labels:
    val = st.number_input(label, value=0.0, step=0.1)
    user_cont.append(val)

st.subheader("🔀 Binary features")
user_bin = []
for i, label in enumerate(binary_labels):
    options = label_encoders[i].classes_.tolist()
    val = st.selectbox(label, options=options, key=f"binary_{i}")
    encoded_val = label_encoders[i].transform([val])[0]
    user_bin.append(encoded_val)

st.subheader("📊 Ordinal features")
user_ord = []
for label in ordinal_labels:
    val = st.number_input(label, value=0.0, step=0.1)
    user_ord.append(val)

# Predict
if st.button("Predict"):
    # Preprocess
    cont_scaled = scaler_cont.transform([user_cont])
    ord_scaled = scaler_ord.transform([user_ord])
    bin_array = np.array(user_bin).reshape(1, -1)
    model_input = np.hstack([cont_scaled, bin_array, ord_scaled])

    # Predict
    input_tensor = torch.tensor(model_input, dtype=torch.float32)
    with torch.no_grad():
        output = kan_model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        prob = torch.softmax(output, dim=1).numpy()[0]

    # Display
    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    if pred == 1:
        st.error(f"⚠️ SAD Detected")
    else:
        st.success(f"✅ No SAD Detected")

    st.write(f"**Probability:**")
    st.write(f"- No SAD: {prob[0]:.2f}")
    st.write(f"- SAD: {prob[1]:.2f}")
