import streamlit as st
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
from pykan.kan import KAN
from pykan.kan.spline import coef2curve

# ========== Utility Functions ==========

def get_layer_components(layer):
    return layer.grid.detach(), layer.coef.detach(), layer.scale_base.detach(), layer.scale_sp.detach()

def compute_spline_outputs(x, grid, coef, k):
    batch_size, input_dim = x.shape
    outputs = []
    for i in range(input_dim):
        xi = x[:, i].unsqueeze(1)
        gi = grid[i]
        ci = coef[i]

        xi_exp = xi.expand(batch_size, k)
        gi_exp = gi.unsqueeze(0).expand(batch_size, k)

        basis = torch.clamp(1 - torch.abs((xi_exp - gi_exp) / (gi[1] - gi[0])), 0, 1)
        yi = basis * ci
        out = yi.sum(dim=1, keepdim=True)
        outputs.append(out)

    return torch.cat(outputs, dim=1)

def compute_combined_output(base_out, spline_out, scale_base, scale_sp):
    return scale_base * base_out + scale_sp * spline_out

def compute_output_second_layer(x, layer, base_fun, k):
    base_out = base_fun(x)
    grid, coef, scale_base, scale_sp = get_layer_components(layer)
    spline_out = compute_spline_outputs(x, grid, coef, k)
    return compute_combined_output(base_out, spline_out, scale_base, scale_sp)

def plot_local_feature_importance(contributions, feature_names):
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
    for bar in bars_top:
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

def get_feature_min_max(original_df, continuous_indices, binary_indices, ordinal_indices):
    min_max_map = {}
    processed_index = 0
    for orig_idx in continuous_indices:
        col = original_df.columns[orig_idx]
        min_max_map[processed_index] = (original_df[col].min(), original_df[col].max())
        processed_index += 1
    for _ in binary_indices:
        min_max_map[processed_index] = (0, 1)
        processed_index += 1
    for orig_idx in ordinal_indices:
        col = original_df.columns[orig_idx]
        min_max_map[processed_index] = (original_df[col].min(), original_df[col].max())
        processed_index += 1
    return min_max_map

def explain_spline_minmax(model, l, i, j, feature_min, feature_max, x_norm_val=None, title=None, feature_names=None):
    x_norm = model.spline_preacts[l][:, j, i].detach().cpu().numpy()
    y_vals = model.spline_postacts[l][:, j, i].detach().cpu().numpy()

    x_real_vals = x_norm * (feature_max - feature_min) + feature_min
    sorted_idx = np.argsort(x_real_vals)
    x_real_vals = x_real_vals[sorted_idx]
    y_vals = y_vals[sorted_idx]

    feature_label = feature_names[i] if feature_names and i < len(feature_names) else f"Feature {i}"

    plt.figure(figsize=(8, 5))
    plt.axhspan(0, max(y_vals.max(), 0), color='lightgray', alpha=0.5)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    for k in range(len(x_real_vals) - 1):
        x0, x1 = x_real_vals[k], x_real_vals[k + 1]
        y0, y1 = y_vals[k], y_vals[k + 1]
        if y0 >= 0 and y1 >= 0:
            plt.plot([x0, x1], [y0, y1], color='blue')
        elif y0 < 0 and y1 < 0:
            plt.plot([x0, x1], [y0, y1], color='orange')
        else:
            zero_x = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
            if y0 < 0:
                plt.plot([x0, zero_x], [y0, 0], color='orange')
                plt.plot([zero_x, x1], [0, y1], color='blue')
            else:
                plt.plot([x0, zero_x], [y0, 0], color='blue')
                plt.plot([zero_x, x1], [0, y1], color='orange')

    plt.xlabel(f"{feature_label} (real units)")
    plt.ylabel("Activation output")
    plt.title(title or f"Spline explanation (Layer {l}, Input {i} → Output {j})")

    if x_norm_val is not None:
        x_real = x_norm_val * (feature_max - feature_min) + feature_min
        plt.axvline(x_real, color='red', linestyle='--', label=f'{feature_label} = {x_real:.2f}')
        if np.min(x_real_vals) <= x_real <= np.max(x_real_vals):
            y_at_x_real = np.interp(x_real, x_real_vals, y_vals)
            plt.plot(x_real, y_at_x_real, 'ro')
        else:
            print(f"x = {x_real:.2f} is outside the spline range.")

    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()

# ========== Streamlit App ==========
st.set_page_config(layout="wide")
st.title("🧠 Predict SAD from MIMIC-IV")
st.markdown("This app predicts SAD and explains spline activations for a standard patient.")

# Load model
model = KAN(width=[53, 1, 2], grid=5, k=3, seed=42)
checkpoint = torch.load('model_with_scores.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.feature_scores = checkpoint['feature_scores']
model.eval()

# Load data artifacts
with open("models/feature_config.pkl", "rb") as f:
    feature_config = pickle.load(f)
with open("models/original_df.pkl", "rb") as f:
    original_df = pickle.load(f)

feature_names = feature_config["continuous_labels"] + feature_config["binary_labels"] + feature_config["ordinal_labels"]
feature_bounds = get_feature_min_max(
    original_df,
    feature_config["original_continuous_indices"],
    feature_config["original_binary_indices"],
    feature_config["original_ordinal_indices"]
)

# Standard patient tensor
standard_tensor = torch.tensor([
    0.4941, 0.1310, 0.5806, 0.6543, 0.4667, 0.7600, 0.1872, 0.1105, 0.1205,
    0.1879, 0.4000, 0.2505, 0.0385, 0.0101, 0.1212, 0.5000, 0.4146, 0.2192,
    0.2308, 0.4692, 0.1370, 0.0220, 0.0287, 0.0802, 0.5870, 0.2391, 1.0000,
    0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0667
])

# Run prediction
with torch.no_grad():
    output = model(standard_tensor.unsqueeze(0))
    pred = torch.argmax(output, dim=1).item()
    prob = torch.softmax(output, dim=1).numpy()[0]

# Show prediction result
st.subheader("🔍 Prediction Result")
if pred == 1:
    st.error("⚠️ SAD Detected")
else:
    st.success("✅ No SAD Detected")
st.write(f"**Probability - No SAD**: {output[0][0]}")
st.write(f"**Probability - SAD**: {output[0][1]}")

# Show global explanation
st.markdown("---")
st.subheader("Global Feature Importance")
df_importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_scores.detach().numpy()
})

# Sort by importance descending
df = df_importances.sort_values(by='Importance', ascending=False)

# Streamlit app
st.title("Feature Importances")

st.write("The following bar chart shows the feature importances ranked from most to least contribution.")

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(df['Feature'], df['Importance'])
ax.invert_yaxis()  # Highest importance at the top
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")
ax.set_title("Feature Importance Ranking")

st.pyplot(fig)

# Show local explanation
st.markdown("---")
st.subheader("📊 Local Feature Importance")
layer1 = model.act_fun[0]
grid1, coef1, scale_base1, scale_sp1 = get_layer_components(layer1)
spline_out1 = compute_spline_outputs(standard_tensor.unsqueeze(0), grid1, coef1, model.k)
base_out1 = layer1.base_fun(standard_tensor.unsqueeze(0))
combined1 = compute_combined_output(base_out1, spline_out1, scale_base1, scale_sp1)
plot_local_feature_importance(combined1[0].detach().numpy(), feature_names)

# Spline activation explanation
st.markdown("---")
st.subheader("🔬 Inspect Individual Spline Contribution")
with st.expander("Show Spline Activation Plot"):
    feature_idx = st.number_input("Choose input feature index", min_value=0, max_value=len(feature_names)-1, step=1)
    layer_output_idx = st.number_input("Hidden neuron index (usually 0)", min_value=0, max_value=0, value=0)
    x_val = standard_tensor[feature_idx].item()
    feature_min, feature_max = feature_bounds[feature_idx]

    if st.button("Plot Spline Contribution"):
        explain_spline_minmax(
            model, l=0, i=feature_idx, j=layer_output_idx,
            feature_min=feature_min, feature_max=feature_max,
            x_norm_val=x_val, feature_names=feature_names
        )