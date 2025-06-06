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
st.title("DelierAlert")
st.markdown("Vroegtijdige waarschuwing voor sepsis-gassocieerd delier")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
}
</style>
''', unsafe_allow_html=True)

# ========== Performance, Output, Data ==========

st.markdown(
    """
    <style>
        .non-link-text {
            color: #3685eb;
            text-decoration: none;
            pointer-events: none;
        }
        .highlight-orange {
            color: #faa63e;
            text-decoration: none;
            pointer-events: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns([2, 3, 3])

with col1:
    st.markdown(
        """
        <div style="background-color:#F0F2F6; padding:20px; border-radius:10px">
            <div style="font-size:20px; font-weight:600;">
                Advies: <span style="color:#3685eb;">Geen SAD</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.toggle("Toon uitleg output", key="toggle_output"):
        st.markdown(
            """
            **Scope van de output**
            Dit model voorspelt of een IC-patiënt wél of géén risico loopt op het ontwikkelen van Sepsis-geassocieerd delier (SAD). De output is binair: risico of geen risico. Het model detecteert geen delier, maar signaleert patiënten die mogelijk extra aandacht nodig hebben.

            **Gebruik van de output**
            De voorspelling is bedoeld als hulpmiddel voor vroegtijdig ingrijpen. Wordt een patiënt als risicogeval gemarkeerd, dan moet hij of zij actief worden gemonitord en overwogen voor preventieve maatregelen.
            Deze tool ondersteunt IC-personeel bij het vroegtijdig herkennen van risicopatiënten. Het neemt het klinisch oordeel niet over, maar helpt om prioriteiten te stellen in de zorg.
            """
        )

with col2:
    with st.container():
        st.markdown(
            """
            <div style="background-color:#F0F2F6; padding:20px; border-radius:10px">
                <div style="font-size:20px; font-weight:600;">
                    Zekerheid: <span style="color:#faa63e;">erg laag</span>
                    <span title="De mate van onzekerheid in de voorspelling van dit model (erg laag, laag, gemiddeld, hoog, erg hoog) op basis van de waarschijnlijkheidsverdeling over de mogelijke uitkomsten."
                        style="cursor: help; margin-left: 8px;">ℹ️</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

with col3:
    with st.container():
        st.markdown(
        """
        <div style="
            background-color: #F0F2F6;
            padding: 16px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            width: 100%;
        ">
            <table style="border-collapse: collapse; width: 100%; border: none; table-layout: fixed;">
                <tr>
                    <th style="text-align: left; padding: 4px 8px; font-weight: 600; border: none;">Patient ID</th>
                    <th style="text-align: left; padding: 4px 8px; font-weight: 600; border: none;">Leeftijd</th>
                    <th style="text-align: left; padding: 4px 8px; font-weight: 600; border: none;">Geslacht</th>
                    <th style="text-align: left; padding: 4px 8px; font-weight: 600; border: none;">ICU Type</th>
                </tr>
                <tr>
                    <td style="padding: 4px 8px; border: none;">2</td>               
                    <td style="padding: 4px 8px; border: none;">60</td>
                    <td style="padding: 4px 8px; border: none;">Man</td>
                    <td style="padding: 4px 8px; border: none;">MICU/SICU</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.toggle("Toon training data bron", key="toggle_data"):
        st.markdown(
            """
            **Data bron:** MIMIC-IV (2024)

            Kenmerken (zoals labwaarden) van **7837** patiënten die zijn opgenomen op de intensivecareafdeling van het Beth Israel Deaconess Medical Center in Boston, Massachusetts.""")
        
        st.markdown("**Labelverdeling trainingsset:**")

        st.markdown(
        """
        <table style="text-align: left;">
            <tr><th>SAD</th><th>Geen SAD</th></tr>
            <tr><td>4705</td><td>3231</td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )
        st.markdown(
            """
            **Leeftijd**

                <45                 772
                45-59               1574
                60-74               2727
                75+                 2764
            **Geslacht**
                
                Vrouw               57.8%         
                Man                 42.2%   
            **Afkomst**

                Europees/Westers    65.7%
                Afrikaans           8.4%
                Latijn-Amerikaans   4.0%
                Aziatisch           2.9%
                Anders of onbekend  19.0%
            **ICU Type**

                MICU                22.8%
                MICU/SICU           19.1%
                CVICU               18.5%
                SICU                13.6%
                TSICU               11.2%
                CCU                 11.1%
                NICU                3.7%
            """
    )

# ========== Local Explanation ==========

# Define columns outside the expanders
column1, column2 = st.columns([1.2, 1.2])

# Column 1 content
with column1:
    with st.expander("ℹ️ **Belang van kenmerken voor individuele voorspelling**"):
        st.write("Deze grafiek laat zien hoe verschillende patiëntkenmerken bijdragen aan de individuele voorspelling van het risico op sepsis-geassocieerde delier (SAD).")
        st.markdown("- **Bovenste grafiek**: Toont de optelsom van alle bijdragen van kenmerken. De balk geeft de som van bijdragen die het risico op SAD verhogen en de rode stippellijn duidt de drempelwaarde aan tussen wel of geen SAD. In dit geval is de som **0.4507**, hoger dan de drempelwaarde van **0.2361**, dus is het advies: *Geen SAD*")
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
        st.markdown("- Het paarse vlak toont het gebied waar binnen de waarde moet blijven om hetzelfde advies te behouden.")
        st.markdown("- De rode stippellijn markeert de waarde voor de huidige patiënt.")
        st.markdown("- Het bijbehorende punt toont hoe sterk deze specifieke waarde het advies beïnvloedt (positief of negatief).")
        st.markdown("- **Blauwe punten** geven waarden die de kans op SAD **verlagen**.")
        st.markdown("- **Oranje punten** geven waarden die de kans op SAD **verhogen**.")

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
    - De temperatuur moet tussen 33.85 °C en 42.00 °C blijven om dezelfde uitkomst te behouden.
    - De glucosewaarde moet tussen 624.40 mg/dL en 8.00 mg/dL blijven om dezelfde uitkomst te behouden.
    - Het natriumgehalte moet tussen 95.00 mEq/L en 145.70 blijven om dezelfde uitkomst te behouden.
    - De INR moet tussen 0.80 en 1.19 blijven om dezelfde uitkomst te behouden.
    - Zolang de patiënt geen mechanische ventilatie ontvangt, blijft de uitkomst ongewijzigd; als dat wel het geval is, verandert het advies naar SAD.
    - Zolang de patiënt geen beroerte heeft gehad, blijft het advies ongewijzigd; als dat wel het geval is, verandert het advies naar SAD.
    - Zolang de patiënt niet is opgenomen op de NICU, blijft het advies ongewijzigd; als dat wel het geval is, verandert het advies naar SAD.
    - De GCS-score moet 1 t/m 5 of 14 of 15 zijn om dezelfde uitkomst te behouden. 
    - De SOFA-score moet 1 t/m 10 zijn om dezelfde uitkomst te behouden.
        """)