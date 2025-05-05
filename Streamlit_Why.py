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

# ========== Load Feature Info ==========
with open("models/feature_config.pkl", "rb") as f:
    feature_config = pickle.load(f)

feature_names = feature_config["continuous_labels"] + feature_config["binary_labels"] + feature_config["ordinal_labels"]

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

    feature_list = [
    "Protrombinetijd (s)",
    "INR",
    "Temperatuur (Celcius)",
    "Mechanische ventilatie n (%)",
    "GCS",
    "SOFA",
    "ICU Type: NICU",
    "Beroerte n (%)",
    "Magnesium (mg/dL)",
    "AKI n (%)",
    "Natrium (mEq/L)",
    "SpO2",
    "Creatinine (mg/dL)",
    "Sedatie n (%)",
    "Vasopressor n (%)",
    "CRRT n (%)",
    "BUN (mg/dL)",
    "Witte bloedcellen (k/uL)",
    "Afkomst: Anders",
    "ICU Type: CVICU",
    "Hartslag (Slagen per Minuut)",
    "PTT (s)",
    "ICU Type: CCU",
    "Ademhalingsfrequentie",
    "COPD n (%)",
    "ICU Type: MICU",
    "Gewicht (Kg)",
    "Leeftijd",
    "Chloride (mEq/L)",
    "ICU Type: SICU",
    "Diastolische bloeddruk (mmHg)",
    "Hemoglobine (g/dL)",
    "Bicarbonaat (mEq/L)",
    "Fosfaat (mg/dL)",
    "Systolische bloeddruk (mmHg)",
    "Afkomst: Aziatisch",
    "Hypertensie n (%)",
    "Gemiddelde arteriele druk (mmHg)",
    "Afkomst: Onbekend",
    "Afkomst: Afrikaans",
    "Totale calcium (mg/dL)",
    "Bloedplaatjes (k/uL)",
    "Anion gap (mEq/L)",
    "Afkomst: Latijns-amerikaans",
    "Geslacht (Male)",
    "Kalium (mEq/L)",
    "ICU Type: MICU/SICU",
    "ICU Type: TSICU",
    "Afkomst: Europees/Westers",
    "AMI n (%)",
    "Diabetes n (%)",
    "Glucose (mg/dL)",
    "CKD n (%)"
]

    selected_label = st.selectbox(
        "Selecteer een kenmerk om de invloed op het advies te bekijken",
        feature_list
    )
    feature_idx = feature_names.index(selected_label)
    
    # Show spline activation for selected feature (layer 0)
    img_path = f"static/local_splines/layer0_input{feature_idx}_to_output0.png"
    st.image(img_path, use_container_width=True)

    
    col2a, col2b = st.columns(2)
    with col2a:
        img_path1 = "static/local_splines/layer1_input0_to_output0.png"
        st.image(img_path1, use_container_width=True)
    with col2b:
        img_path2 = "static/local_splines/layer1_input0_to_output1.png"
        st.image(img_path2, use_container_width=True)