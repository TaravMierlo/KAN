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
st.title("DelierAlert")
st.markdown("Vroegtijdige waarschuwing voor sepsis-gassocieerd delier")

# ========== Load Feature Info ==========
with open("models/feature_config.pkl", "rb") as f:
    feature_config = pickle.load(f)

feature_names = feature_config["continuous_labels"] + feature_config["binary_labels"] + feature_config["ordinal_labels"]

# Ensure correct number of features
assert len(feature_names) == 53, f"Expected 53 features, found {len(feature_names)}."

df_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_scores.detach().numpy()
})
df_sorted = df_importances.sort_values(by='Importance', ascending=False)

# ========== Top Metrics Section ==========

prediction = "SAD Gedetecteerd"
delta = -0.31
num_features = 53

col1, col2, col3 = st.columns([1,1,2])

with col1:
    color = "red" if "SAD" in prediction else "green"
    st.markdown(f"""
    <style>
    .tooltip {{
        position: relative;
        display: inline-block;
        vertical-align: middle;
        cursor: pointer;
    }}

    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 200px;
        background-color: #fff;
        color: #555;
        text-align: left;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.85em;
    }}

    .tooltip .tooltiptext::after {{
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #555 transparent transparent transparent;
    }}

    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>

    <div style="border: 1px solid #ccc; padding: 1em; border-radius: 10px; background-color: #f9f9f9; text-align: left;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <h4 style="margin: 0;">Advies: {'SAD' if 'SAD' in prediction else 'Geen SAD'}</h4>
            <span class="tooltip">
                <span style="font-weight: bold; border: 1px solid #888; border-radius: 50%; padding: 0 6px; background: #eee;">i</span>
                <span class="tooltiptext">this is a test</span>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


with col2:
    st.markdown(f"""
    <div style="border: 1px solid #ccc; padding: 1em; border-radius: 10px; background-color: #f9f9f9; text-align: left;">
        <h4>Zekerheid: laag</h4>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="border: 1px solid #ccc; padding: 1em; border-radius: 10px; background-color: #f9f9f9; text-align: left;">
        <h4>Data</h4>
        <p style="font-size: 24px; font-weight: bold;">{num_features}</p>
        <p style="font-size: 14px;">Kenmerken gebruikt in voorspelling</p>
    </div>
    """, unsafe_allow_html=True)

# ========== Global Explanation ==========
st.markdown("---")
st.subheader("Hoe werkt het model?")

col1, col2 = st.columns([1, 0.8])

with col1:
    with st.expander("ℹ️ **Ranglijst van Kenmerkbelang**"):
        st.write("Hieronder zie je hoe belangrijk elk kenmerk gemiddeld genomen is in het hele model. "
        "Deze ranglijst geeft een algemeen beeld van welke gegevens het meest bijdragen aan het advies.")

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.barh(df_sorted['Feature'], df_sorted['Importance'], color='#3685eb')
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

with col2:
    with st.expander("ℹ️ **Invloed per kenmerk**"):
        st.write("Deze grafieken tonen hoe verschillende klinische variabelen bijdragen aan het voorspelde risico op sepsis-geassocieerde delier.")
        st.markdown("- X-as: Waarden van een specifieke variabele.")
        st.markdown("- Y-as: De invloed van deze waarde op de voorspelling van het model. Hoe verder een punt van nul ligt (positief of negatief), hoe groter de impact op het advies.")
        st.markdown("- Oranje punten duiden op waarden die het voorspelde risico op SAD **verhogen**.")
        st.markdown("- Blauwe punten duiden op waarden die het voorspelde risico **verlagen**.")

        st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
        }
        </style>
        ''', unsafe_allow_html=True)

    
    selected_label = st.selectbox(
        "Selecteer een kenmerk om de invloed op het advies te bekijken",
        df_sorted['Feature'].tolist()
    )
    feature_idx = feature_names.index(selected_label)
    
    # Show spline activation for selected feature (layer 0)
    img_path = f"static/splines/layer0_input{feature_idx}_to_output0.png"
    st.image(img_path, use_container_width=True)

    # Show two additional images side-by-side (layer 1)
    with st.expander("ℹ️ **Uitkomst Advies**"):
        st.write("Deze grafieken tonen hoe de som van de activatiefunctie-uitkomsten van alle variabelen wordt leidt tot de ruwe modeluitkomst, die bepaalt of het advies **SAD** of **geen SAD** is.")
        st.markdown("- De **linkse grafiek** hoort bij het advies *Geen SAD*: de ruwe modeluitkomst stijgt naarmate de gecombineerde activatie toeneemt.")
        st.markdown("- De **rechtse grafiek** hoort bij het advies *SAD*: de ruwe modeluitkomst daalt naarmate de gecombineerde activatie toeneemt.")
        st.markdown("- X-as: Som van activatiefuncties. De totale invloed van alle klinische variabelen.")
        st.markdown("- Y-as: Ruwe modeluitkomst voor de uiteindelijke classificatie.")

        st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
        }
        </style>
        ''', unsafe_allow_html=True)
    
    col2a, col2b = st.columns(2)
    with col2a:
        img_path1 = "static/splines/layer1_input0_to_output0.png"
        st.image(img_path1, use_container_width=True)
    with col2b:
        img_path2 = "static/splines/layer1_input0_to_output1.png"
        st.image(img_path2, use_container_width=True)