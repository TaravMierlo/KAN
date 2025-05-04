import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

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

# Dummy importance scores for display purposes
importance_scores = list(range(len(feature_names), 0, -1))
df_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance_scores
})
df_sorted = df_importances.sort_values(by='Importance', ascending=False)

# ========== Static Prediction Display ==========
st.subheader("🔍 Prediction Result")
st.error("⚠️ SAD Detected")
st.write("**Probability - No SAD**: 0.31")
st.write("**Probability - SAD**: 0.69")

# ========== Global Explanation ==========
st.markdown("---")
st.subheader("🌍 Global Feature Importance")

col1, col2 = st.columns([1, 1])

with col1:
    st.write("**Feature Importance Ranking**")
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.barh(df_sorted['Feature'], df_sorted['Importance'])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

with col2:
    selected_label = st.selectbox(
        "Select a feature to inspect spline activation",
        df_sorted['Feature'].tolist()
    )
    feature_idx = feature_names.index(selected_label)
    img_path = f"static/splines/layer0_input{feature_idx}_to_output0.png"
    st.image(img_path, caption=f"Spline activation for {selected_label}", use_column_width=True)

# ========== Local Explanation ==========
st.markdown("---")
st.subheader("📊 Local Feature Importance")
st.image("static/local_feature_importance_waterfall.png", caption="Feature contributions for the prediction", use_column_width=True)
