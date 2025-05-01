# streamlit_app.py

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
from pykan.kan import KAN
from torch.serialization import add_safe_globals

# ========== Local Feature Importance Helpers ==========
def plot_local_feature_importance(contributions, feature_names):
    import matplotlib.pyplot as plt

    sorted_indices = np.argsort(np.abs(contributions))[::-1]
    sorted_contributions = contributions[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    colors = ['blue' if val >= 0 else 'orange' for val in sorted_contributions]

    total_for = np.sum([c for c in contributions if c > 0])
    total_against = np.sum([-c for c in contributions if c < 0])

    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 6])

    ax1 = fig.add_subplot(gs[0])
    bars_top = ax1.barh(["Bewijs SAD", "Bewijs geen SAD"], [total_for, total_against], color=["blue", "orange"])
    ax1.set_xlim(0, max(total_for, total_against) * 1.2)
    ax1.set_title("Totaal Bewijs Voor en Tegen SAD")

    for i, bar in enumerate(bars_top):
        x_val = bar.get_width()
        ax1.text(x_val + 0.01, bar.get_y() + bar.get_height() / 2, f"{x_val:.2f}", va='center', fontsize=8)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = fig.add_subplot(gs[1])
    bars = ax2.barh(range(len(sorted_contributions)), np.abs(sorted_contributions), color=colors)
    ax2.set_yticks(range(len(sorted_feature_names)))
    ax2.set_yticklabels(sorted_feature_names)
    ax2.set_title("Patientkenmerken gesorteerd op belangrijkheid voor advies")
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.invert_yaxis()

    for i, bar in enumerate(bars):
        x_val = bar.get_width()
        sign = "+" if sorted_contributions[i] >= 0 else "-"
        ax2.text(x_val + 0.005, i, f"{sign}{x_val:.2f}", va='center', fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def get_layer_components(layer):
    return layer.grid.detach(), layer.coef.detach(), layer.scale_base.detach(), layer.scale_sp.detach()


def compute_spline_outputs(x, grid, coef, k=None, plot=False, layer_idx=0):
    batch_size, input_dim = x.shape
    outputs = []

    for i in range(input_dim):
        xi = x[:, i].unsqueeze(1)  # shape (batch_size, 1)
        gi = grid[i]               # shape (n_knots,)
        ci = coef[i]               # shape (n_knots,)

        n_knots = gi.shape[0]
        assert ci.shape[0] == n_knots, f"Grid and coef size mismatch at feature {i}"

        xi_exp = xi.expand(batch_size, n_knots)
        gi_exp = gi.unsqueeze(0).expand(batch_size, n_knots)

        denom = gi[1] - gi[0] if n_knots > 1 else 1.0  # Avoid division by zero
        basis = torch.clamp(1 - torch.abs((xi_exp - gi_exp) / denom), 0, 1)
        yi = basis * ci  # shape (batch_size, n_knots)
        out = yi.sum(dim=1, keepdim=True)  # shape (batch_size, 1)

        outputs.append(out)

    return torch.cat(outputs, dim=1)  # shape (batch_size, input_dim)

def compute_combined_output(base_out, spline_out, scale_base, scale_sp):
    return scale_base * base_out + scale_sp * spline_out


def compute_output_second_layer(x, layer, base_fun, k):
    base_out = base_fun(x)
    grid, coef, scale_base, scale_sp = get_layer_components(layer)
    spline_out = compute_spline_outputs(x, grid, coef, k)
    combined = compute_combined_output(base_out, spline_out, scale_base, scale_sp)
    return combined


def manual_forward_kan(model, x_input, feature_names):
    if x_input.dim() == 1:
        x_input = x_input.unsqueeze(0)

    # First Layer
    layer1 = model.act_fun[0]
    grid1, coef1, scale_base1, scale_sp1 = get_layer_components(layer1)

    spline_out1 = compute_spline_outputs(x_input, grid1, coef1, model.k)
    base_out1 = layer1.base_fun(x_input)
    combined1 = compute_combined_output(base_out1, spline_out1, scale_base1, scale_sp1)
    layer1_out = combined1.sum(dim=1, keepdim=True)

    plot_local_feature_importance(combined1[0].detach().numpy(), feature_names)

    # Second Layer
    layer2 = model.act_fun[1]
    out = compute_output_second_layer(layer1_out, layer2, layer2.base_fun, model.k)

    pred_class = torch.argmax(out, dim=1).item()
    return out, pred_class

# ========== Streamlit UI ==========
st.set_page_config(layout="wide")
st.title("🧠 Predict SAD from MIMIC-IV")
st.markdown("This app predicts SAD and visualizes local feature contributions for a standard patient.")

# ========== Load artifacts ==========
st.markdown("### 🔧 Loading preprocessing artifacts")
try:
    with open("models/feature_config.pkl", "rb") as f:
        feature_config = pickle.load(f)
    st.success("Loaded feature configuration")
except Exception as e:
    st.error(f"Error loading feature configuration: {e}")

# ========== Load model ==========
st.markdown("### 🤖 Loading KAN model")
model = KAN(width=[53, 1, 2], grid=5, k=3, seed=42, device=None)

try:
    state_dict = torch.load("kan_model.pt")
    model.load_state_dict(state_dict)
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

model.eval()

# ========== Standard Patient ==========
st.markdown("### 🧍 Standard Patient Input")
standard_tensor = torch.tensor([
    0.4941, 0.1310, 0.5806, 0.6543, 0.4667, 0.7600, 0.1872, 0.1105, 0.1205,
    0.1879, 0.4000, 0.2505, 0.0385, 0.0101, 0.1212, 0.5000, 0.4146, 0.2192,
    0.2308, 0.4692, 0.1370, 0.0220, 0.0287, 0.0802, 0.5870, 0.2391, 1.0000,
    0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0667
])

feature_names = feature_config["continuous_labels"] + feature_config["binary_labels"] + feature_config["ordinal_labels"]

if st.button("Predict and Show Local Explanation"):
    try:
        output, pred_class = manual_forward_kan(model, standard_tensor, feature_names)
        prob = torch.softmax(output, dim=1).numpy()[0]

        st.markdown("### 🔍 Prediction Result")
        if pred_class == 1:
            st.error("⚠️ SAD Detected")
        else:
            st.success("✅ No SAD Detected")

        st.write("**Probability:**")
        st.write(f"- No SAD: {prob[0]:.2f}")
        st.write(f"- SAD: {prob[1]:.2f}")

        st.markdown("### 📊 Local Feature Contributions")
        st.markdown("Plotted above ⬆")

    except Exception as e:
        st.error(f"Prediction or plotting failed: {e}")
