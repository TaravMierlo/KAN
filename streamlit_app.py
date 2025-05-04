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

# ========== Load Feature Info ==========
with open("models/feature_config.pkl", "rb") as f:
    feature_config = pickle.load(f)

feature_names = feature_config["continuous_labels"] + feature_config["binary_labels"] + feature_config["ordinal_labels"]

# Ensure correct number of features
assert len(feature_names) == 53, f"Expected 53 features, found {len(feature_names)}."

# Global Explanation (Side by Side)
st.markdown("---")
st.subheader("🌍 Global Feature Importance")

df_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_scores.detach().numpy()
})
df_sorted = df_importances.sort_values(by='Importance', ascending=False)

# ========== Static Prediction Display ==========
st.subheader("🔍 Prediction Result")
st.error("⚠️ SAD Detected")
st.write("**Probability - No SAD**: 0.31")
st.write("**Probability - SAD**: 0.69")

# ========== Global Explanation ==========
st.markdown("---")
st.subheader("Hoe werkt het model?")

col1, col2 = st.columns([1, 1])

with col1:
    st.write("**Ranglijst van Kenmerkbelang**")
    st.markdown(
        "Hieronder zie je hoe belangrijk elk kenmerk gemiddeld genomen is in het hele model. "
        "Deze ranglijst geeft een algemeen beeld van welke gegevens het meest bijdragen aan de voorspelling."
    )
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.barh(df_sorted['Feature'], df_sorted['Importance'])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

with col2:
    st.write("**Invloed per kenmerk**")
    st.markdown(
        "Hier zie je hoe het model een kenmerk omzet via een activatiefunctie." \
        "Deze functie is geleerd tijdens het trainen en bepaalt welk signaal een bepaalde invoerwaarde bijdraagt aan de uiteindelijke voorspelling."
    )
    
    selected_label = st.selectbox(
        "Selecteer een kenmerk om de invloed op het advies te bekijken",
        df_sorted['Feature'].tolist()
    )
    feature_idx = feature_names.index(selected_label)
    
    # Show spline activation for selected feature (layer 0)
    img_path = f"static/splines/layer0_input{feature_idx}_to_output0.png"
    st.image(img_path, use_container_width=True)

    # Show two additional images side-by-side (layer 1)
    st.write("**Layer 1 Output Splines**")
    st.markdown("De outputs van alle activatiefuncties worden bij elkaar opgeteld tot één totaalscore. " \
    "Dit is de basis voor de eindbeslissing van het model.")
    
    col2a, col2b = st.columns(2)
    with col2a:
        img_path1 = "static/splines/layer1_input0_to_output0.png"
        st.image(img_path1, use_container_width=True)
    with col2b:
        img_path2 = "static/splines/layer1_input0_to_output1.png"
        st.image(img_path2, use_container_width=True)
    st.markdown("Op basis van de totaalscore kiest het model tussen twee uitkomsten: SAD of Geen SAD. De lijnen laten zien hoe de uiteindelijke beslissing verandert afhankelijk van de som van alle activatiefunctie-uitkomsten.")

# ========== Local Explanation ==========
st.markdown("---")
st.subheader("📊 Local Feature Importance")
st.image("static/local_feature_importance_waterfall.png", caption="Feature contributions for the prediction", use_container_width=True)
