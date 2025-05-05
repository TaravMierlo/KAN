import streamlit as st
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pykan.kan import KAN


# Load model
model = KAN(width=[53, 1, 2], grid=5, k=3, seed=42)
checkpoint = torch.load('model_with_scores.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.feature_scores = checkpoint['feature_scores']
model.spline_preacts = checkpoint['model_spline_preacts']
model.spline_postacts = checkpoint['model_spline_postacts']
model.eval()

# Load data
with open("models/original_df.pkl", "rb") as f:
    original_df = pickle.load(f)

# ========== Streamlit Config ==========
st.set_page_config(layout="wide")
st.title("🧠 Predict SAD from MIMIC-IV")
st.markdown("This app predicts SAD and explains spline activations for a standard patient. (Prototype Version)")

# ========== Local Explanation ==========
st.markdown("---")

st.subheader("📊 Local Feature Importance")

# Define columns outside the expanders
column1, column2 = st.columns([1, 1])

# Then use each column
with column1:
    st.write("ℹ️ **Belang van kenmerken voor specifiek advies**")
    st.image("static/local_feature_importance_waterfall.png", use_container_width=True)

with column2:
    st.write("ℹ️ **Ranglijst van Kenmerkbelang**")
    st.image("static/global_feature_importance_bar.png", use_container_width=True)  # Placeholder


