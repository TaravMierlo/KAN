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
            Dit model voorspelt of een IC-patiënt wel of geen risico loopt op het ontwikkelen van Sepsis-geassocieerd delier (SAD). De output is binair: risico of geen risico. Het model detecteert geen delier, maar signaleert patiënten die mogelijk extra aandacht nodig hebben.

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
                    Certainty: <span style="color:#faa63e;">very low</span>
                    <span title="The level of uncertainty in the prediction of this model (very low, low, medium, high, very high) based on the probability distribution over the possible outcomes."
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
                    <th style="text-align: left; padding: 4px 8px; font-weight: 600; border: none;">Age</th>
                    <th style="text-align: left; padding: 4px 8px; font-weight: 600; border: none;">Gender</th>
                    <th style="text-align: left; padding: 4px 8px; font-weight: 600; border: none;">ICU Type</th>
                </tr>
                <tr>
                    <td style="padding: 4px 8px; border: none;">2</td>               
                    <td style="padding: 4px 8px; border: none;">60</td>
                    <td style="padding: 4px 8px; border: none;">Male</td>
                    <td style="padding: 4px 8px; border: none;">MICU/SICU</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.toggle("Show training data source", key="toggle_data"):
        st.markdown(
            """
            **Data source:** MIMIC-IV (2024)

            Characteristics (such as lab values) of **7837** patients who were admitted to the intensive care unit of the Beth Israel Deaconess Medical Center in Boston, Massachusetts.""")
        
        st.markdown("**Labelverdeling trainingsset:**")

        st.markdown(
        """
        <table style="text-align: left;">
            <tr><th>SAD</th><th>No SAD</th></tr>
            <tr><td>4705</td><td>3231</td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )
        st.markdown(
            """
            **Age**

                <45                 772
                45-59               1574
                60-74               2727
                75+                 2764
            **Gender**
                
                Female              57.8%         
                Male                42.2%   
            **Race**

                European/Caucasian  65.7%
                Afrikan             8.4%
                Latin-American      4.0%
                Asian               2.9%
                Other or unknown    19.0%
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
st.subheader("")

# Define columns outside the expanders
column1, column2 = st.columns([1.5, 1])

# Then use each column
with column1:
    with st.expander("ℹ️ **Belang van kenmerken voor individuele voorspelling**"):
        st.write("Deze grafiek laat zien hoe verschillende patiëntkenmerken bijdragen aan de individuele voorspelling van het risico op sepsis-geassocieerde delier (SAD).")
        st.markdown("- **Bovenste grafiek**: Toont de optelsom van alle bijdragen van kenmerken. De balk geeft de som van bijdragen die het risico op SAD verhogen en de rode stippellijn duidt de drempelwaarde aan tussen wel of geen SAD. In dit geval is de som **0.4507**, hoger dan de drempelwaarde **0.2361**, dus is het advies: *Geen SAD*")
        st.markdown("- **Onderste grafiek**: Visualiseert de individuele bijdragen van kenmerken aan de voorspelling.")
        st.markdown("- Oranje balken duiden op kenmerken die de kans op SAD **verhogen**.")
        st.markdown("- Blauwe balken duiden op kenmerken die de kans op SAD **verlagen**.")
        st.markdown("- De lengte van de balk geeft de mate van invloed aan; langere balken wijzen op een sterkere bijdrage aan de voorspelling.")

    st.image("static/local_feature_importance_waterfall.png", use_container_width=True)

with column2:
    with st.expander("ℹ️ **Effect van individuele variabele op het advies**"):
        st.write("Deze grafieken tonen aan hoe de waarde van patiëntkenmerken (hier: Protrombinetijd) bijdragen aan het advies.")
        st.markdown("- **X-as**: Waarden die het kenmerk kan aannemen.")
        st.markdown("- **Y-as**: De bijdrage van elke waarde aan het uiteindelijke advies.")
        st.markdown("- De rode stippellijn markeert de waarde voor de huidige patiënt.")
        st.markdown("- Het bijbehorende punt toont hoe sterk deze specifieke waarde het advies beïnvloedt (positief of negatief).")
        st.markdown("- **Blauwe punten** geven waarden die de kans op SAD **verlagen**.")
        st.markdown("- **Oranje punten** geven waarden die de kans op SAD **verhogen**.")

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

    with st.expander("ℹ️ **Uitleg uitkomst van het model**"):
        st.write("Deze grafiek laat zien hoe het model tot een advies komt op basis van de som van alle bijdragen.")
        st.markdown("- **X-as**: De som van bijdragen van alle individuele patiëntkenmerken.")
        st.markdown("- **Y-as**: De ruwe modeluitkomst voordat het wordt omgezet naar een advies (SAD of Geen SAD).")
        st.markdown("- De **blauwe lijn** toont de curve voor het advies *Geen SAD*.")
        st.markdown("- De **oranje lijn** toont de curve voor het advies *SAD*.")
        st.markdown("- De **rode stippellijn** markeert de inputwaarde van deze specifieke patiënt.")
        st.markdown("- De **groene stippellijn** toont de beslissingsgrens: ligt de groene lijn rechts hiervan, dan is het advies *Geen SAD*; links is het advies *SAD*.")
        st.markdown("- De gemarkeerde waarden op de lijnen geven de ruwe modeloutput bij de inputwaarde.")

    img_path1 = "static/local_splines/layer1_input0_adviesuitkomst.png"
    st.image(img_path1, use_container_width=True)

    
