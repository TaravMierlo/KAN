import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle
from pykan.kan import KAN
from pykan.kan.MultKAN import MultKAN
from torch.serialization import add_safe_globals

# =========================
# Load preprocessing artifacts
# =========================
st.title("🧠 Predict SAD from MIMIC-IV")
st.markdown("### 🔧 Loading preprocessing artifacts")

try:
    with open("models/scaler_cont.pkl", "rb") as f:
        scaler_cont = pickle.load(f)
    st.success("Loaded scaler for continuous features")

    with open("models/scaler_ord.pkl", "rb") as f:
        scaler_ord = pickle.load(f)
    st.success("Loaded scaler for ordinal features")

    with open("models/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    st.success("Loaded label encoders for binary features")

    with open("models/feature_config.pkl", "rb") as f:
        feature_config = pickle.load(f)
    st.success("Loaded feature configuration")
except Exception as e:
    st.error(f"Error loading preprocessing files: {e}")

continuous_labels = feature_config["continuous_labels"]
binary_labels = feature_config["binary_labels"]
ordinal_labels = feature_config["ordinal_labels"]

# =========================
# Load trained KAN model
# =========================
st.markdown("### 🤖 Loading KAN model")

model = KAN(
    width=[53, 1, 2], grid=5, k=3,
    seed=42, device=None
)

try:
    state_dict = torch.load("kan_model.pt")
    model.load_state_dict(state_dict)
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

model.eval()

# =========================
# Streamlit UI
# =========================
st.markdown("### 🧾 Input Features")
st.markdown("Fill in the patient features below to predict SAD.")

st.subheader("🧪 Continuous features")
user_cont = []
for label in continuous_labels:
    val = st.number_input(label, value=0.0, step=0.1)
    user_cont.append(val)
st.write("**Raw continuous input:**", user_cont)

st.subheader("🔀 Binary features")
user_bin = []
for i, label in enumerate(binary_labels):
    options = label_encoders[i].classes_.tolist()
    val = st.selectbox(label, options=options, key=f"binary_{i}")
    encoded_val = label_encoders[i].transform([val])[0]
    user_bin.append(encoded_val)
st.write("**Encoded binary input:**", user_bin)

st.subheader("📊 Ordinal features")
user_ord = []
for label in ordinal_labels:
    val = st.number_input(label, value=0.0, step=0.1)
    user_ord.append(val)
st.write("**Raw ordinal input:**", user_ord)

# =========================
# Prediction logic
# =========================
if st.button("Predict"):
    try:
        # Preprocessing
        st.markdown("### 🧼 Preprocessing Inputs")
        cont_scaled = scaler_cont.transform([user_cont])
        ord_scaled = scaler_ord.transform([user_ord])
        bin_array = np.array(user_bin).reshape(1, -1)

        st.write("**Scaled continuous:**", cont_scaled)
        st.write("**Scaled ordinal:**", ord_scaled)
        st.write("**Reshaped binary:**", bin_array)

        model_input = np.hstack([cont_scaled, bin_array, ord_scaled])
        st.write("**Final model input:**")
        st.code(model_input)

        # Model prediction
        input_tensor = torch.tensor(model_input, dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            prob = torch.softmax(output, dim=1).numpy()[0]

        # Output results
        st.markdown("---")
        st.subheader("🔍 Prediction Result")
        if pred == 1:
            st.error("⚠️ SAD Detected")
        else:
            st.success("✅ No SAD Detected")

        st.write("**Probability:**")
        st.write(f"- No SAD: {prob[0]:.2f}")
        st.write(f"- SAD: {prob[1]:.2f}")

        st.markdown("---")
        st.subheader("📤 Raw Model Output")
        st.code(output.numpy())

    except Exception as e:
        st.error(f"Prediction failed: {e}")

        # Feature Importance
        # =========================
        st.markdown("### 🧠 Feature Importance")
        try:
            import matplotlib.pyplot as plt

            scores = model.feature_score.detach().numpy()
            all_labels = continuous_labels + binary_labels + ordinal_labels

            fig, ax = plt.subplots()
            ax.barh(all_labels, scores)
            ax.set_xlabel("Importance Score")
            ax.set_title("Feature Importance (from KAN)")

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error displaying feature importance: {e}")