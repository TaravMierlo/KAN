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

# ========== Global Explanation ==========
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
        st.write("Deze grafieken tonen voor iedere patiëntkenmerk hoe deze bijdraagt aan het voorspelde risico op sepsis-geassocieerde delier.")
        st.markdown("- X-as: De waarden voor een patiëntkenmerk.")
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

    st.markdown ("""
        Voor elk van de 53 kenmerken wordt de bijdrage aan de uitkomst bepaald via een zogenoemde spline-functie. De eerste grafiek toont dit voor Protrombinetijd. Al deze bijdragen worden opgeteld om een wwrde te krijgen per patiënt. Deze som wordt ingevoerd in twee nieuwe functies (voor iedere categorie een) die bepaalt of de ruwe model uitkomst boven of onder de drempel ligt. Op basis daarvan geeft het model zijn advies (*SAD* of *geen SAD*).
                 """)

    # Show two additional images side-by-side (layer 1)
    with st.expander("ℹ️ **Uitkomst Advies**"):
        st.write("Deze grafiek toont hoe de som van de activatiefunctie-uitkomsten van alle variabelen leidt tot de ruwe modeluitkomst, die bepaalt of het advies **SAD** of **geen SAD** is.")
        st.markdown("- De **blauwe lijn** (*Geen SAD*): de ruwe modeluitkomst stijgt naarmate de gecombineerde activatie toeneemt.")
        st.markdown("- De **oranje lijn** (*SAD*): de ruwe modeluitkomst daalt naarmate de gecombineerde activatie toeneemt.")
        st.markdown("- X-as: Som van activatiefuncties. De totale invloed van alle klinische variabelen.")
        st.markdown("- Y-as: Ruwe modeluitkomst voor de uiteindelijke classificatie.")
        st.markdown("- Groene gestippelde lijn (*Beslissingsgrens: 0.2361*): Als de som van de activatie outputs groter is dan deze waarde (ongeveer 0.2361), kiest het model voor ""Geen SAD"", anders voor ""SAD"".")

        st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
        }
        </style>
        ''', unsafe_allow_html=True)
    
    img_path = "static/splines/advies_uitkomst_plot.png"
    st.image(img_path, use_container_width=True)