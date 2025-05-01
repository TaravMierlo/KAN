import streamlit as st
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from pykan.kan import KAN
from your_module import coef2curve, explain_spline_minmax  # Replace with correct import if needed

# ============== Page Setup ==============
st.set_page_config(page_title="SAD Prediction with KAN", layout="wide")
st.title("🧠 Predict SAD from MIMIC-IV")
st.markdown("### 🔧 Loading preprocessing artifacts")

# ============== Load Preprocessing Artifacts ==============
scaler_cont = pickle.load(open("models/scaler_cont.pkl", "rb"))
scaler_ord = pickle.load(open("models/scaler_ord.pkl", "rb"))
label_encoders = pickle.load(open("models/label_encoders.pkl", "rb"))
feature_config = pickle.load(open("models/feature_config.pkl", "rb"))

continuous_labels = feature_config["continuous_labels"]
binary_labels = feature_config["binary_labels"]
ordinal_labels = feature_config["ordinal_labels"]
feature_names = continuous_labels + binary_labels + ordinal_labels

# ============== Load Model ==============
model = KAN(width=[53, 1, 2], grid=5, k=3, seed=42)
model.load_state_dict(torch.load("kan_model.pt", map_location=torch.device("cpu")))
model.eval()

# ============== Fixed Input Tensor ==============
x_input = torch.tensor([
    0.4941, 0.1310, 0.5806, 0.6543, 0.4667, 0.7600, 0.1872, 0.1105, 0.1205,
    0.1879, 0.4000, 0.2505, 0.0385, 0.0101, 0.1212, 0.5000, 0.4146, 0.2192,
    0.2308, 0.4692, 0.1370, 0.0220, 0.0287, 0.0802, 0.5870, 0.2391, 1.0000,
    0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0667
], dtype=torch.float32)

st.markdown("### 📥 Fixed Input Tensor")
st.code(x_input, language="python")

# ============== Main Button ==============
if st.button("🔍 Run Explanation and Prediction"):
    with st.spinner("Running forward pass..."):
        try:
            out, pred_class = manual_forward_kan(model, x_input, feature_names, splineplots=False, detailed_computation=True)
            st.success(f"✅ Prediction: {pred_class} | Raw Output: {out.detach().numpy()}")
        except Exception as e:
            st.error(f"Error during computation: {e}")

# ============== Helper Functions ==============
def manual_forward_kan(model, x_input, feature_names, splineplots=False, detailed_computation=False):
    if x_input.dim() == 1:
        x_input = x_input.unsqueeze(0)

    # First Layer
    layer1 = model.act_fun[0]
    grid1, coef1, scale_base1, scale_sp1 = layer1.grid, layer1.coef, layer1.scale_base, layer1.scale_sp

    spline_out1 = compute_spline_outputs(x_input, grid1, coef1, model.k)
    base_out1 = layer1.base_fun(x_input)
    combined1 = compute_combined_output(base_out1, spline_out1, scale_base1, scale_sp1)
    layer1_out = combined1.sum(dim=1, keepdim=True)

    if detailed_computation:
        print_local_contributions_streamlit(combined1, base_out1, spline_out1, scale_base1, scale_sp1, feature_names)

    plot_local_feature_importance_streamlit(combined1[0].detach().numpy(), feature_names)

    # Second Layer
    layer2 = model.act_fun[1]
    out = compute_output_second_layer(layer1_out, layer2, layer2.base_fun, model.k)
    pred_class = torch.argmax(out, dim=1).item()

    return out, pred_class

def compute_spline_outputs(x_input, grid, coef, k):
    spline_outputs = []
    for i in range(x_input.shape[1]):
        xi = x_input[:, [i]]
        grid_i = grid[i].unsqueeze(0)
        coef_i = coef[i].unsqueeze(0)
        spline_i = coef2curve(xi, grid_i, coef_i, k)[:, 0, 0]
        spline_outputs.append(spline_i)
    return torch.stack(spline_outputs, dim=1)

def compute_combined_output(base_out, spline_out, scale_base, scale_sp):
    return scale_base.T * base_out + scale_sp.T * spline_out

def compute_output_second_layer(layer_input, layer, base_fun, k):
    grid, coef, scale_base, scale_sp = layer.grid, layer.coef, layer.scale_base, layer.scale_sp
    spline_out = coef2curve(layer_input, grid, coef, k)[:, 0, :]
    base_out = base_fun(layer_input)
    return scale_base * base_out + scale_sp * spline_out

def print_local_contributions_streamlit(combined, base_out, spline_out, scale_base, scale_sp, feature_names):
    st.markdown("### 🔍 Detailed Contribution Breakdown")
    rows = []
    for j in range(len(feature_names)):
        sb = scale_base[j].item()
        ss = scale_sp[j].item()
        base_val = base_out[0, j].item()
        spline_val = spline_out[0, j].item()
        combined_val = combined[0, j].item()
        rows.append(f"{feature_names[j]} = {sb:.4f} * {base_val:.4f} + {ss:.4f} * {spline_val:.4f} = {combined_val:.4f}")
    st.text("\n".join(rows))

def plot_local_feature_importance_streamlit(contributions, feature_names):
    sorted_indices = np.argsort(np.abs(contributions))[::-1]
    sorted_contributions = contributions[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    colors = ['blue' if val >= 0 else 'orange' for val in sorted_contributions]

    total_for = np.sum([c for c in contributions if c > 0])
    total_against = np.sum([-c for c in contributions if c < 0])

    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 6])

    ax1 = fig.add_subplot(gs[0])
    ax1.barh(["SAD", "No SAD"], [total_for, total_against], color=["blue", "orange"])
    ax1.set_title("Evidence For and Against SAD")
    ax1.set_xlim(0, max(total_for, total_against) * 1.2)

    ax2 = fig.add_subplot(gs[1])
    bars = ax2.barh(range(len(sorted_contributions)), np.abs(sorted_contributions), color=colors)
    ax2.set_yticks(range(len(sorted_feature_names)))
    ax2.set_yticklabels(sorted_feature_names)
    ax2.set_title("Feature Importance (Sorted)")
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.invert_yaxis()

    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()
