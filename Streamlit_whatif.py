import streamlit as st
st.set_page_config(layout="wide")

import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from pykan.kan import KAN
from pykan.kan.spline import *
import sys

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

# Get the feature groups
continuous_labels = feature_config["continuous_labels"]
binary_labels = feature_config["binary_labels"]
ordinal_labels = feature_config["ordinal_labels"]

# Concatenate the full feature list
feature_names = continuous_labels + binary_labels + ordinal_labels

# Compute index ranges
continuous_indices = list(range(len(continuous_labels)))
binary_indices = list(range(len(continuous_labels), len(continuous_labels) + len(binary_labels)))
ordinal_indices = list(range(len(continuous_labels) + len(binary_labels), len(feature_names)))

# Filter features based on available spline image
valid_features = []
for idx, name in enumerate(feature_names):
    img_path = f"static/cf_splines_htbt/layer0_input{idx}_to_output0.png"
    if os.path.exists(img_path):
        valid_features.append((idx, name))

# Load data
with open("models/original_df.pkl", "rb") as f:
    original_df = pickle.load(f)

with open("models/scaler_cont.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/scaler_ord.pkl", "rb") as f:
    ordinal_encoder = pickle.load(f)

# ========== Streamlit Config ==========

st.title("🧠 Predict SAD from MIMIC-IV")
st.markdown("This app predicts SAD and explains spline activations for a standard patient. (Prototype Version)")

st.markdown('''
<style>
[data-testid="stMarkdownContainer"] ul{
    list-style-position: inside;
}
</style>
''', unsafe_allow_html=True)

# Inject CSS to make a sticky sidebar-style box
# Use raw HTML + CSS for sticky layout

orange = '#faa63e'
blue = '#3685eb' 

# ========== Load Manual Forward KAN ==========

# Prevent Streamlit from inspecting torch.classes
sys.modules['torch.classes'].__path__ = []

def get_feature_min_max(original_df, continuous_indices, binary_indices, ordinal_indices):
    """
    Returns a dictionary mapping feature index in processed_data to (min, max).
    """
    min_max_map = {}
    processed_index = 0

    # Continuous features
    for orig_idx in continuous_indices:
        col = original_df.columns[orig_idx]
        col_min = original_df[col].min()
        col_max = original_df[col].max()
        min_max_map[processed_index] = (col_min, col_max)
        processed_index += 1

    # Binary features (fixed range 0 to 1)
    for _ in binary_indices:
        min_max_map[processed_index] = (0, 1)
        processed_index += 1

    # Ordinal features
    for orig_idx in ordinal_indices:
        col = original_df.columns[orig_idx]
        col_min = original_df[col].min()
        col_max = original_df[col].max()
        min_max_map[processed_index] = (col_min, col_max)
        processed_index += 1

    return min_max_map

cached_preacts = model.spline_preacts
cached_postacts = model.spline_postacts

def explain_spline_minmax(
    model, l, i, j, feature_min, feature_max,
    x_norm_val=None, title=None, feature_names=None,
    save_dir=None
):

    x_norm = cached_preacts[l][:, j, i].detach().cpu().numpy()
    y_vals = cached_postacts[l][:, j, i].detach().cpu().numpy()

    x_real_vals = x_norm * (feature_max - feature_min) + feature_min
    sorted_idx = np.argsort(x_real_vals)
    x_real_vals = x_real_vals[sorted_idx]
    y_vals = y_vals[sorted_idx]

    # Use the feature name if provided, else fallback to index
    feature_label = feature_names[i] if feature_names is not None and i < len(feature_names) else f"Feature {i}"

    plt.figure(figsize=(8, 5))

    # Fill area above y = 0 with gray
    plt.axhspan(0, max(y_vals.max(), 0), color='lightgray', alpha=0.5)

    # Draw horizontal line at y = 0
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Plot each spline segment with color based on sign
    for k in range(len(x_real_vals) - 1):
        x0, x1 = x_real_vals[k], x_real_vals[k + 1]
        y0, y1 = y_vals[k], y_vals[k + 1]

        if y0 >= 0 and y1 >= 0:
            plt.plot([x0, x1], [y0, y1], color=blue)
        elif y0 < 0 and y1 < 0:
            plt.plot([x0, x1], [y0, y1], color=orange)
        else:
            # Interpolate zero crossing
            zero_x = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
            if y0 < 0:
                plt.plot([x0, zero_x], [y0, 0], color=orange)
                plt.plot([zero_x, x1], [0, y1], color=blue)
            else:
                plt.plot([x0, zero_x], [y0, 0], color=blue)
                plt.plot([zero_x, x1], [0, y1], color=orange)

    # Plot dots at data points, color-coded by sign
    for x, y in zip(x_real_vals, y_vals):
        color = blue if y >= 0 else orange
        plt.plot(x, y, 'o', color=color, markersize=4)

    # Optional: vertical line and red dot for current input value
    if x_norm_val is not None:
        x_real = x_norm_val * (feature_max - feature_min) + feature_min
        plt.axvline(x_real, color='red', linestyle='--', label=f'{feature_label} = {x_real:.2f}')
        if np.min(x_real_vals) <= x_real <= np.max(x_real_vals):
            y_at_x_real = np.interp(x_real, x_real_vals, y_vals)
            plt.plot(x_real, y_at_x_real, 'ro')
            plt.text(
                x_real, y_at_x_real,
                f"{y_at_x_real:.4f}",
                fontsize=9,
                ha='left',
                va='bottom',
                color='red',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='red', facecolor='white', alpha=0.7)
            )
            print(f"At {feature_label} = {x_real:.2f}, Activation output = {y_at_x_real:.4f}")
        else:
            print(f"x = {x_real:.2f} is outside the spline range.")

    plt.xlabel(f"{feature_label}")
    plt.ylabel("Activation output")
    plt.title(title or f"Invloed van {feature_label} op het advies")
    plt.legend()
    plt.grid(True)

    # Save the plot if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"layer{l}_input{i}_to_output{j}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Saved plot to {filepath}")

    #plt.show()
    st.pyplot(plt.gcf())

def get_layer_components(layer):
    return layer.grid, layer.coef, layer.scale_base, layer.scale_sp

def compute_spline_outputs(x_input, grid, coef, k, splineplots=False, layer_idx=0):
    spline_outputs = []
    for i in range(x_input.shape[1]):
        xi = x_input[:, [i]]
        grid_i = grid[i].unsqueeze(0)
        coef_i = coef[i].unsqueeze(0)

        spline_i = coef2curve(xi, grid_i, coef_i, k)[:, 0, 0]
        spline_outputs.append(spline_i)

        if splineplots:
            feature_bounds = get_feature_min_max(original_df, continuous_indices, binary_indices, ordinal_indices)
            f_min, f_max = feature_bounds[i]
            explain_spline_minmax(model, layer_idx, i, 0, f_min, f_max, x_norm_val=xi.item(), title=None, feature_names=original_df.columns.tolist()[:-1], save_dir="static/local_splines")

    return torch.stack(spline_outputs, dim=1)

def compute_combined_output(base_out, spline_out, scale_base, scale_sp):
    return scale_base.T * base_out + scale_sp.T * spline_out

def print_local_contributions(combined, base_out, spline_out, scale_base, scale_sp, feature_names):
    print(f"\nDetailed computation for patient:\n")

    contributions = []
    for j in range(len(feature_names)):
        sb = scale_base[j].item()
        ss = scale_sp[j].item()
        base_val = base_out[0, j].item()
        spline_val = spline_out[0, j].item()
        combined_val = combined[0, j].item()

        contributions.append((combined_val, feature_names[j], sb, base_val, ss, spline_val))

    # Sort by combined_val descending
    contributions.sort(reverse=True, key=lambda x: abs(x[0]))

    # Print sorted contributions
    for combined_val, name, sb, base_val, ss, spline_val in contributions:
        print(f"{name} = {sb:.4f} * {base_val:.4f} + {ss:.4f} * {spline_val:.4f} = {combined_val:.4f}")


def plot_local_feature_importance(contributions, feature_names, real_values=None, activation_flip_point=0.2361, sum_range=(-1.5, 1.7)):
    sad_color = orange
    geen_sad_color = blue

    # Sort features by absolute importance
    sorted_indices = np.argsort(np.abs(contributions))[::-1]
    sorted_contributions = contributions[sorted_indices]

    if real_values is not None:
        sorted_feature_names = [f"{feature_names[i]}: {real_values[i]:.2f}" for i in sorted_indices]
    else:
        sorted_feature_names = [feature_names[i] for i in sorted_indices]

    colors = [geen_sad_color if val >= 0 else sad_color for val in sorted_contributions]

    # Total sum of contributions
    total_sum = np.sum(contributions)
    epsilon = 1e-8
    is_geen_sad = total_sum > activation_flip_point + epsilon
    prediction = "Geen SAD" if is_geen_sad else "SAD"
    margin = activation_flip_point - total_sum
    margin_text = (
        f"Nog {margin:.3f} nodig voor Geen SAD" if not is_geen_sad
        else f"{abs(margin):.3f} boven de drempel"
    )

    # Create figure and layout
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 6])

    # === Top Plot: Activatiebalk tov drempel ===
    ax1 = fig.add_subplot(gs[0])
    x_min, x_max = sum_range
    ax1.set_xlim(x_min, x_max)

    # Background shading
    ax1.axvspan(x_min, activation_flip_point, color=sad_color, alpha=0.04, zorder=1)
    ax1.axvspan(activation_flip_point, x_max, color=geen_sad_color, alpha=0.04, zorder=1)

    # Decision threshold line
    ax1.axvline(activation_flip_point, color='red', linestyle='--', label=f"Drempel = {activation_flip_point:.4f}", zorder=2)

    # Draw bar from threshold to total_sum
    bar_color = geen_sad_color if is_geen_sad else sad_color
    if total_sum >= activation_flip_point:
        bar_start = activation_flip_point
        bar_width = total_sum - activation_flip_point
    else:
        bar_start = total_sum
        bar_width = activation_flip_point - total_sum
    ax1.barh(0, width=bar_width, left=bar_start, color=bar_color, height=0.2, zorder=3)

    # Clean look
    ax1.set_ylim(-0.6, 0.6)
    ax1.set_yticks([])
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Prediction and sum text
    ax1.set_title(f"Voorspelling: {prediction}  ({margin_text})",
                  fontsize=13, fontweight='bold',
                  color=blue if is_geen_sad else orange, pad=15)
    ax1.text(total_sum, 0.3, f"Som = {total_sum:.4f}", fontsize=9, ha='center', zorder=4)
    ax1.legend(loc='upper right')

    # === Bottom Plot: Lokale bijdragen ===
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.barh(range(len(sorted_contributions)), np.abs(sorted_contributions), color=colors)
    ax2.set_yticks(range(len(sorted_feature_names)))
    ax2.set_yticklabels(sorted_feature_names)
    ax2.set_title("Bijdragen van Patiëntkenmerken aan Advies")
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.invert_yaxis()

    for i, bar in enumerate(bars):
        x_val = bar.get_width()
        sign = "+" if sorted_contributions[i] >= 0 else "-"
        ax2.text(x_val + 0.005, i, f"{sign}{x_val:.2f}", va='center', fontsize=8)

    plt.tight_layout()
    st.pyplot(plt.gcf())

def compute_output_second_layer(layer_input, layer, base_fun, k):
    grid, coef, scale_base, scale_sp = get_layer_components(layer)
    spline_out = coef2curve(layer_input, grid, coef, k)[:, 0, :]
    base_out = base_fun(layer_input)
    return scale_base * base_out + scale_sp * spline_out

def denormalize_instance(normalized_tensor, scaler, ordinal_encoder, continuous_indices, ordinal_indices, binary_indices):
    """
    Reconstructs the full true feature vector from a normalized patient tensor, denormalizing continuous
    and ordinal features and keeping binary features unchanged.

    Parameters:
    - normalized_tensor (torch.Tensor): The normalized patient data (1D tensor).
    - scaler (MinMaxScaler): Fitted scaler for continuous features.
    - ordinal_encoder (MinMaxScaler): Fitted scaler for ordinal features.
    - continuous_indices (list): Indices of continuous features in full vector.
    - ordinal_indices (list): Indices of ordinal features in full vector.
    - binary_indices (list): Indices of binary features in full vector.

    Returns:
    - torch.Tensor: 1D tensor with all features in original (true) form.
    """
    if hasattr(normalized_tensor, "detach"):
        normalized_tensor = normalized_tensor.detach().cpu().numpy()
    
    # Prepare output array
    full_true_values = np.zeros_like(normalized_tensor)

    # Denormalize continuous
    continuous_normalized = normalized_tensor[continuous_indices].reshape(1, -1)
    full_true_values[continuous_indices] = scaler.inverse_transform(continuous_normalized).flatten()

    # Denormalize ordinal
    ordinal_normalized = normalized_tensor[ordinal_indices].reshape(1, -1)
    full_true_values[ordinal_indices] = ordinal_encoder.inverse_transform(ordinal_normalized).flatten()

    # Keep binary features unchanged
    full_true_values[binary_indices] = normalized_tensor[binary_indices]

    return torch.tensor(full_true_values, dtype=torch.float32)

def explain_spline_output(
    model, l, i, title="Advies uitkomst", feature_names=None,
    save_dir="static/local_splines", combined=None
):
    # Extract activations for both classes
    x0 = cached_preacts[l][:, 0, i].detach().cpu().numpy()
    y0 = cached_postacts[l][:, 0, i].detach().cpu().numpy()

    x1 = cached_preacts[l][:, 1, i].detach().cpu().numpy()
    y1 = cached_postacts[l][:, 1, i].detach().cpu().numpy()

    # Sort both on x-values
    sort_idx = np.argsort(x0)
    x_vals = x0[sort_idx]
    y0 = y0[sort_idx]
    y1 = y1[sort_idx]  # Assume x0 == x1 due to shared inputs

    # Decision boundary: where the curves intersect
    diff = y0 - y1
    sign_change = np.where(np.diff(np.sign(diff)))[0]

    x_flip = None
    if len(sign_change) > 0:
        idx = sign_change[0]
        x_flip = np.interp(0, [diff[idx], diff[idx + 1]], [x_vals[idx], x_vals[idx + 1]])

    # Create directory
    os.makedirs(save_dir, exist_ok=True)

    # Plot both curves
    plt.figure()
    plt.plot(x_vals, y0, label='Geen SAD')
    plt.plot(x_vals, y1, label='SAD')

    # Plot decision boundary
    if x_flip is not None:
        plt.axvline(x=x_flip, color='red', linestyle='dashed', label=f'Beslissingsgrens: {x_flip:.4f}')

    # Plot combined input if provided
    if combined is not None:
        combined_val = float(combined)  # Ensure scalar
        plt.axvline(x=combined_val, color='purple', linestyle='dotted', label=f'Som Activatie Functies: {combined_val:.4f}')

        # Interpolate y values on both curves
        y0_interp = np.interp(combined_val, x_vals, y0)
        y1_interp = np.interp(combined_val, x_vals, y1)

        # Plot intersection points
        plt.scatter([combined_val], [y0_interp], color=blue)
        plt.scatter([combined_val], [y1_interp], color=orange)

        # Annotate the y-values
        plt.text(combined_val, y0_interp, f'{y0_interp:.2f}', color=blue, va='bottom', ha='right')
        plt.text(combined_val, y1_interp, f'{y1_interp:.2f}', color=orange, va='top', ha='right')

    plt.xlabel('Som Activatie Functies Outputs')
    plt.ylabel('Ruwe Uitkomst Model')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Save
    filename = f"layer{l}_input{i}_adviesuitkomst.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Grafiek opgeslagen als: {filepath}")

def manual_forward_kan(model, x_input, splineplots=False, detailed_computation=False):
    if x_input.dim() == 1:
        x_input = x_input.unsqueeze(0)

    feature_names = original_df.columns.tolist()[:-1] # Do not include SAD
    batch_size = x_input.shape[0]

    # First Layer
    layer1 = model.act_fun[0]
    grid1, coef1, scale_base1, scale_sp1 = get_layer_components(layer1)

    spline_out1 = compute_spline_outputs(x_input, grid1, coef1, model.k, splineplots, layer_idx=0)
    base_out1 = layer1.base_fun(x_input)
    combined1 = compute_combined_output(base_out1, spline_out1, scale_base1, scale_sp1)
    layer1_out = combined1.sum(dim=1, keepdim=True)

    if detailed_computation == True:
        print_local_contributions(combined1, base_out1, spline_out1, scale_base1, scale_sp1, feature_names)

    real_values_tensor = denormalize_instance(
    x_input[0],
    scaler,
    ordinal_encoder,
    continuous_indices,
    ordinal_indices,
    binary_indices
    )
    
    real_values = real_values_tensor.numpy()
    
    plot_local_feature_importance(combined1[0].detach().numpy(), feature_names, real_values=real_values)

    # Second Layer
    #explain_spline_output(model, 1, 0, combined=layer1_out.detach().numpy())
    
    layer2 = model.act_fun[1]
    out = compute_output_second_layer(layer1_out, layer2, layer2.base_fun, model.k)

    print("Final output:", out)
    pred_class = torch.argmax(out, dim=1).item()
    print(f"Prediction: {pred_class}")

    return out, pred_class

def streamlit_what_if_widget(
    normalized_tensor,
    model,
    manual_forward_kan,
    scaler,
    ordinal_encoder,
    continuous_indices,
    ordinal_indices,
    binary_indices,
    feature_names
):

    st.subheader("Wat als Scenario: Pas waarden aan")

    # Step 1: Denormalize original input
    full_true_tensor = denormalize_instance(
        normalized_tensor,
        scaler,
        ordinal_encoder,
        continuous_indices,
        ordinal_indices,
        binary_indices
    )
    full_true_values = full_true_tensor.numpy()
    editable_indices = continuous_indices + ordinal_indices

    # Step 2: Compute contributions to determine slider order
    with torch.no_grad():
        layer = model.act_fun[0]
        grid, coef, scale_base, scale_sp = get_layer_components(layer)
        spline_out = compute_spline_outputs(normalized_tensor.unsqueeze(0), grid, coef, model.k)
        base_out = layer.base_fun(normalized_tensor.unsqueeze(0))
        combined = compute_combined_output(base_out, spline_out, scale_base, scale_sp)
        contributions = combined[0].detach().numpy()
    sorted_indices = sorted(editable_indices, key=lambda i: abs(contributions[i]), reverse=True)

    # Step 3: Build sliders
    st.markdown("Pas de waarden van de kenmerken aan en bekijk het effect op het advies.")
    modified_values = full_true_values.copy()

    for i in sorted_indices:
        fname = feature_names[i]
        if i in continuous_indices:
            min_val = scaler.data_min_[continuous_indices.index(i)]
            max_val = scaler.data_max_[continuous_indices.index(i)]
            modified_values[i] = st.slider(
                label=fname,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(full_true_values[i]),
                step=0.1,
                key=f"slider_{i}"
            )
        elif i in ordinal_indices:
            min_val = ordinal_encoder.data_min_[ordinal_indices.index(i)]
            max_val = ordinal_encoder.data_max_[ordinal_indices.index(i)]
            modified_values[i] = st.slider(
                label=fname,
                min_value=int(round(min_val)),
                max_value=int(round(max_val)),
                value=int(round(full_true_values[i])),
                step=1,
                key=f"slider_{i}"
            )

    # Step 4: Run button
    if st.button("🚀 Run What-If Analysis"):
        cont_vals = np.array([modified_values[i] for i in continuous_indices]).reshape(1, -1)
        ord_vals = np.array([modified_values[i] for i in ordinal_indices]).reshape(1, -1)
        modified_norm = modified_values.copy()
        modified_norm[continuous_indices] = scaler.transform(cont_vals).flatten()
        modified_norm[ordinal_indices] = ordinal_encoder.transform(ord_vals).flatten()
        modified_tensor = torch.tensor(modified_norm, dtype=torch.float32)

        st.success("Voorspelling gestart met aangepaste waarden.")
        out, pred_class = manual_forward_kan(model, modified_tensor.unsqueeze(0), detailed_computation=True)


# ========== Local Explanation ==========
patient2 = torch.tensor([0.4941, 0.1310, 0.5806, 0.6543, 0.4667, 0.7600, 0.1872, 0.1105, 0.1205,
        0.1879, 0.4000, 0.2505, 0.0385, 0.0101, 0.1212, 0.5000, 0.4146, 0.2192,
        0.2308, 0.4692, 0.1370, 0.0220, 0.0287, 0.0802, 0.5870, 0.2391, 1.0000,
        0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0667])

# ========== Local Explanation ==========
st.markdown("---")

# Define columns outside the expanders
column1, column2, column3 = st.columns([1.2, 1.2, 1.2])

with column1:
    st.subheader("Oorspronkelijke Advies")

    with st.expander("ℹ️ **Belang van kenmerken voor individuele voorspelling**"):
        st.write("Deze grafiek laat zien hoe verschillende patiëntkenmerken bijdragen aan de individuele voorspelling van het risico op sepsis-geassocieerde delier (SAD).")
        st.markdown("- **Bovenste grafiek**: Toont de optelsom van alle bijdragen van kenmerken. De balk geeft de som van bijdragen die het risico op SAD verhogen en de rode stippellijn duidt de drempelwaarde aan tussen wel of geen SAD. In dit geval is de som **0.4507**, voorbij de drempelwaarde van **0.2361**, dus is het advies: *Geen SAD*")
        st.markdown("- **Onderste grafiek**: Visualiseert de individuele bijdragen van kenmerken aan de voorspelling.")
        st.markdown("- Oranje balken duiden op kenmerken die de kans op SAD **verhogen**.")
        st.markdown("- Blauwe balken duiden op kenmerken die de kans op SAD **verlagen**.")
        st.markdown("- De lengte van de balk geeft de mate van invloed aan; langere balken wijzen op een sterkere bijdrage aan de voorspelling.")

    # Call forward to get the plot-ready figure
    out, pred_class = manual_forward_kan(model, patient2)

# Column 2 content
with column2:
    with st.expander("ℹ️ **Effect van individuele variabele op het advies**"):
        st.write("Deze grafieken tonen aan hoe de waarde van patiëntkenmerken (hier: Protrombinetijd) bijdragen aan het advies.")
        st.markdown("- **X-as**: Waarden die het kenmerk kan aannemen.")
        st.markdown("- **Y-as**: De bijdrage (‘activatie output’) van elke waarde aan het uiteindelijke advies.")
        st.markdown("- De rode stippellijn markeert de waarde voor de huidige patiënt.")
        st.markdown("- Het bijbehorende punt toont hoe sterk deze specifieke waarde het advies beïnvloedt (positief of negatief).")
        st.markdown("- **Blauwe punten** geven waarden die het risico op SAD **verlagen**.")
        st.markdown("- **Oranje punten** geven waarden die het risico op SAD **verhogen**.")

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

# Column 3 content
with column3:
    # ========== What-If Scenario ==========

    streamlit_what_if_widget(
        normalized_tensor=patient2,
        model=model,
        manual_forward_kan=manual_forward_kan,
        scaler=scaler,
        ordinal_encoder=ordinal_encoder,
        continuous_indices=continuous_indices,
        ordinal_indices=ordinal_indices,
        binary_indices=binary_indices,
        feature_names=feature_names
    )

    with st.expander("ℹ️ **Bekijk Oorspronkelijke Advies**"):

        # Call forward to get the plot-ready figure
        out, pred_class = manual_forward_kan(model, patient2)