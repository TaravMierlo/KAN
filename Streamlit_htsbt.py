import streamlit as st
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
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

# Filter features based on available spline image
valid_features = []
for idx, name in enumerate(feature_names):
    img_path = f"static/cf_splines_htsbt/layer0_input{idx}_to_output0.png"
    if os.path.exists(img_path):
        valid_features.append((idx, name))

# Load data
with open("models/original_df.pkl", "rb") as f:
    original_df = pickle.load(f)

# ========== Streamlit Config ==========
st.set_page_config(layout="wide")
st.title("🧠 Predict SAD from MIMIC-IV")
st.markdown("This app predicts SAD and explains spline activations for a standard patient. (Prototype Version)")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
}
</style>
''', unsafe_allow_html=True)

# ========== Local Explanation ==========
st.markdown("---")

# Define columns outside the expanders
column1, column2 = st.columns([1.2, 1.2])

# Column 1 content
with column1:
    with st.expander("ℹ️ **Belang van kenmerken voor individuele voorspelling**"):
        st.write("Deze grafiek laat zien hoe verschillende patiëntkenmerken bijdragen aan de individuele voorspelling van het risico op sepsis-geassocieerde delier (SAD).")
        st.markdown("- **Bovenste grafiek**: Toont de optelsom van alle bijdragen van kenmerken. De balk geeft de som van bijdragen die het risico op SAD verhogen en de rode stippellijn duidt de drempelwaarde aan tussen wel of geen SAD. In dit geval is de som **0.4507**, voorbij de drempelwaarde van **0.2361**, dus is het advies: *Geen SAD*")
        st.markdown("- **Onderste grafiek**: Visualiseert de individuele bijdragen van kenmerken aan de voorspelling.")
        st.markdown("- Oranje balken duiden op kenmerken die de kans op SAD **verhogen**.")
        st.markdown("- Blauwe balken duiden op kenmerken die de kans op SAD **verlagen**.")
        st.markdown("- De lengte van de balk geeft de mate van invloed aan; langere balken wijzen op een sterkere bijdrage aan de voorspelling.")

    st.image("static/local_feature_importance_waterfall.png", use_container_width=True)

# Column 2 content
with column2:
    with st.expander("ℹ️ **Effect van individuele variabele op het advies**"):
        st.write("Deze grafieken tonen aan hoe de waarde van patiëntkenmerken (hier: Protrombinetijd) bijdragen aan het advies.")
        st.markdown("- **X-as**: Waarden die het kenmerk kan aannemen.")
        st.markdown("- **Y-as**: De bijdrage (‘activatie output’) van elke waarde aan het uiteindelijke advies.")
        st.markdown("- De paarse stippellijn markeert de maximale verandering van de waarde die mogelijk is om hetzelfde advies te behouden.")
        st.markdown("- De rode stippellijn markeert de waarde voor de huidige patiënt.")
        st.markdown("- Het bijbehorende punt toont hoe sterk deze specifieke waarde het advies beïnvloedt (positief of negatief).")
        st.markdown("- **Blauwe punten** geven waarden die het risico op SAD **verlagen**.")
        st.markdown("- **Oranje punten** geven waarden die het risico op SAD **verhogen**.")

    # Dropdown with only valid features
    selected_idx, selected_label = st.selectbox(
        "Selecteer een kenmerk om te zien in hoeverre het kan veranderen zonder dat het advies verandert",
        valid_features,
        format_func=lambda x: x[1]
    )

    # Show corresponding spline image
    img_path = f"static/cf_splines_htsbt/layer0_input{selected_idx}_to_output0.png"
    st.image(img_path, use_container_width=True)

    st.markdown("#### Uitleg per Kenmerk")
    st.markdown("""
    - De temperatuur mag niet lager worden dan 33.85 °C om hetzelfde advies te behouden.
    - De glucosewaarde mag maximaal 624.40 mg/dL zijn om hetzelfde advies te behouden.
    - Het natriumgehalte mag niet hoger zijn dan 145.70 mEq/L om hetzelfde advies te behouden.
    - De INR mag maximaal 1.19 zijn om hetzelde advies te behouden.
    - Zolang de patiënt geen mechanische ventilatie ontvangt, blijft het advies ongewijzigd; als dat wel het geval is, verandert het advies naar SAD.
    - Zolang de patiënt geen beroerte heeft gehad, blijft het advies ongewijzigd; als dat wel het geval is, verandert het advies naar SAD.
    - Zolang de patiënt niet is opgenomen op de NICU, blijft het advies ongewijzigd; als dat wel het geval is, verandert het advies naar SAD.
    - De GCS-score moet buiten het bereik van 5.14 en 13.90 blijven om dezelfde voorspelling te behouden. Dus lager dan 5.14 of hoger dan 14.00. 
    - De SOFA-score mag niet hoger zijn dan 10.06 om hetzelfde advies te behouden.
        """)