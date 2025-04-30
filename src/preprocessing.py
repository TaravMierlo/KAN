import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_and_preprocess_mimic(path="data/MIMIC-IV.dta", batch_size=512, test_size=0.3, seed=42):
    """
    Loads and preprocesses the MIMIC-IV dataset.
    
    Args:
        path (str): Path to the .dta file.
        batch_size (int): Batch size for DataLoaders.
        test_size (float): Fraction of data to use as test set.
        seed (int): Random seed for reproducibility.
    
    Returns:
        dict: Dictionary containing processed tensors, DataLoaders, original data, and feature labels.
    """
    # Load and clean data
    data = pd.read_stata(path).drop(
        columns=[
            "deliriumtime", "hosp_mort", "icu28dmort", "stay_id",
            "icustay", "hospstay", "sepsistime"
        ]
    ).dropna()

    # One-hot encode categorical variables
    data = data.drop("race", axis=1).join(pd.get_dummies(data["race"]))
    data = data.drop("first_careunit", axis=1).join(pd.get_dummies(data["first_careunit"]))

    # Rename columns
    rename_dict = {
        "age": "Age", "weight": "Weight (Kg)", "gender": "Gender (Male)",
        "temperature": "Temperature (Celcius)", "heart_rate": "Heart Rate (Beats per Minute)",
        "resp_rate": "Respiratory Rate", "sbp": "Systolic BP (mmHG)", "dbp": "Diastolic BP (mmHG)",
        "mbp": "Mean arterial BP (mmHG)", "wbc": "WBC (K/uL)", "hemoglobin": "Hemoglobin (g/dL)",
        "platelet": "Platelet (k/uL)", "bun": "BUN (mg/dL)", "cr": "Creatinine (mg/dL)",
        "glu": "Glucose (mg/dL)", "Na": "Sodium (mEq/L)", "Cl": "Chloride (mEq/L)",
        "K": "Potassium (mEq/L)", "Mg": "Magnesium (mg/dL)", "Ca": "Total calcium (mg/dL)",
        "P": "Phosphate (mg/dL)", "inr": "INR", "pt": "Prothrombin time (s)", "ptt": "PTT (s)",
        "bicarbonate": "Bicarbonate (mEq/L)", "aniongap": "Anion gap (mEq/L)", "gcs": "GCS",
        "vent": "MV n (%)", "crrt": "CRRT n (%)", "vaso": "Vasopressor n (%)", "seda": "Sedation n (%)",
        "sofa_score": "SOFA", "ami": "AMI n (%)", "ckd": "CKD n (%)", "copd": "COPD n (%)",
        "hyperte": "Hypertension n (%)", "dm": "Diabetes n (%)", "sad": "SAD", "aki": "AKI n (%)",
        "stroke": "Stroke n (%)", "AISAN": "Race: Asian", "BLACK": "Race: Black", "HISPANIC": "Race: Hispanic",
        "OTHER": "Race: Other", "WHITE": "Race: White", "unknown": "Race: Unknown",
        "CCU": "ICU Type: CCU", "CVICU": "ICU Type: CVICU", "MICU": "ICU Type: MICU",
        "MICU/SICU": "ICU Type: MICU/SICU", "NICU": "ICU Type: NICU", "SICU": "ICU Type: SICU",
        "TSICU": "ICU Type: TSICU"
    }
    data.rename(columns=rename_dict, inplace=True)

    # Define target and input features
    y = data["SAD"].values
    X = data.drop("SAD", axis=1).values

    # Feature indices
    continuous_indices = list(range(0, 27))
    ordinal_indices = [27, 32]
    binary_indices = list(set(range(X.shape[1])) - set(continuous_indices) - set(ordinal_indices))

    # Scale continuous and ordinal data
    scaler = MinMaxScaler()
    normalized_continuous_data = scaler.fit_transform(X[:, continuous_indices].astype(np.float32))
    encoded_ordinal_data = MinMaxScaler().fit_transform(X[:, ordinal_indices].astype(np.float32))

    # Encode binary features
    binary_data = X[:, binary_indices]
    encoded_binary_data = np.array([
        LabelEncoder().fit_transform(binary_data[:, i]) 
        for i in range(binary_data.shape[1])
    ]).T.astype(np.int64)

    # Concatenate all features
    processed_data = np.hstack([normalized_continuous_data, encoded_binary_data, encoded_ordinal_data])

    # Train-test split
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        processed_data, y, np.arange(len(data)),
        test_size=test_size, random_state=seed, stratify=y
    )

    # Store original feature labels
    column_names = list(data.drop("SAD", axis=1).columns)
    continuous_labels = [column_names[i] for i in continuous_indices]
    binary_labels = [column_names[i] for i in binary_indices]
    ordinal_labels = [column_names[i] for i in ordinal_indices]

    # Store reordered original DataFrame
    reordered_data = data[continuous_labels + binary_labels + ordinal_labels + ["SAD"]]
    original_data = reordered_data.copy()

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=len(X_test), shuffle=False)

    return {
        "train_input": X_train,
        "test_input": X_test,
        "train_label": y_train,
        "test_label": y_test,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "original_data": original_data,
        "continuous_labels": continuous_labels,
        "binary_labels": binary_labels,
        "ordinal_labels": ordinal_labels,
        "train_indices": train_idx,
        "test_indices": test_idx
    }