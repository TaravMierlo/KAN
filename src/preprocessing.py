import torch
import os
import pickle
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

    # Race and first_careunit will be separated into a new boolean 
    #  column for each race or unit,
    dummies = pd.get_dummies(data["race"])
    data = data.drop("race", axis=1).join(dummies)
    dummies = pd.get_dummies(data["first_careunit"])
    data = data.drop("first_careunit", axis=1).join(dummies)

    # Write out abreviations in full and add type of measurement.
    data = data.rename(
        columns={
            "age": "Age",
            "weight": "Weight (Kg)",
            "gender": "Gender (Male)",
            "temperature": "Temperature (Celcius)",
            "heart_rate": "Heart Rate (Beats per Minute)",
            "resp_rate": "Respiratory Rate",
            "sbp": "Systolic BP (mmHG)",
            "dbp": "Diastolic BP (mmHG)",
            "mbp": "Mean arterial BP (mmHG)",
            "wbc": "WBC (K/uL)",
            "hemoglobin": "Hemoglobin (g/dL)",
            "platelet": "Platelet (k/uL)",
            "bun": "BUN (mg/dL)",
            "cr": "Creatinine (mg/dL)",
            "glu": "Glucose (mg/dL)",
            "Na": "Sodium (mEq/L)",
            "Cl": "Chloride (mEq/L)",
            "K": "Potassium (mEq/L)",
            "Mg": "Magnesium (mg/dL)",
            "Ca": "Total calcium (mg/dL)",
            "P": "Phosphate (mg/dL)",
            "inr": "INR",
            "pt": "Prothrombin time (s)",
            "ptt": "PTT (s)",
            "bicarbonate": "Bicarbonate (mEq/L)",
            "aniongap": "Anion gap (mEq/L)",
            "gcs": "GCS",
            "vent": "MV n (%)",
            "crrt": "CRRT n (%)",
            "vaso": "Vasopressor n (%)",
            "seda": "Sedation n (%)",
            "sofa_score": "SOFA",
            "ami": "AMI n (%)",
            "ckd": "CKD n (%)",
            "copd": "COPD n (%)",
            "hyperte": "Hypertension n (%)",
            "dm": "Diabetes n (%)",
            "sad": "SAD",
            "aki": "AKI n (%)",
            "stroke": "Stroke n (%)",
            "AISAN": "Race: Asian",
            "BLACK": "Race: Black",
            "HISPANIC": "Race: Hispanic",
            "OTHER": "Race: Other",
            "WHITE": "Race: White",
            "unknown": "Race: Unknown",
            "CCU": "ICU Type: CCU",
            "CVICU": "ICU Type: CVICU",
            "MICU": "ICU Type: MICU",
            "MICU/SICU": "ICU Type: MICU/SICU",
            "NICU": "ICU Type: NICU",
            "SICU": "ICU Type: SICU",
            "TSICU": "ICU Type: TSICU"
        }
    )
    
    # The SAD column is the target.
    x = data.drop("SAD", axis=1).values
    y = data["SAD"].values

    # continuous features
    continuous_indices = [
        0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
    ]

    # gcs: 27 and sofa score: 32
    ordinal_indices = [27, 32]

    # binary features
    binary_indices = [
        2, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52
    ]

    # Scale continuous and ordinal data
    continuous_data = x[:, continuous_indices].astype(np.float32)
    scaler = MinMaxScaler()
    normalized_continuous_data = scaler.fit_transform(continuous_data)

    # Label encode ordinal data (use min-max scaling for now, 
    #  potentially use one-hot encoding) 
    #  - same as race and first_careunit
    ordinal_data = x[:, ordinal_indices]
    ordinal_encoder = MinMaxScaler()
    encoded_ordinal_data = ordinal_encoder.fit_transform(ordinal_data)

    # Label encode binary data
    binary_data = x[:, binary_indices]
    label_encoders = [LabelEncoder() for _ in binary_indices]
    encoded_categorical_data = np.array(
        [
            le.fit_transform(binary_data[:, i]) 
            for i, le in enumerate(label_encoders)
        ]
    ).T

    encoded_categorical_data = encoded_categorical_data.astype(np.int64)

    # Combine continuous and encoded categorical data
    processed_data = np.hstack((
        normalized_continuous_data, 
        encoded_categorical_data, 
        encoded_ordinal_data
    ))

    X_train, X_test, y_train, y_test, train_indices, test_indices = \
        train_test_split(
            processed_data, 
            y, 
            np.arange(len(data)), 
            random_state=42, 
            test_size=0.3, 
            stratify=y
        )

    # Keep a copy of the original data, 
    # but put it in the same order as the processed data.
    data_columns = list(data.drop("SAD", axis=1).columns.values)
    continuous_labels = [data_columns[i] for i in continuous_indices]
    binary_labels = [data_columns[i] for i in binary_indices]
    ordinal_labels = [data_columns[i] for i in ordinal_indices]
    data = data[[
        *continuous_labels, 
        *binary_labels, 
        *ordinal_labels, 
        "SAD"
    ]]
    original_data = data.copy()  

    # Convert the numpy arrays to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Dataloader for training and testing 
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)

    # DataLoader
    train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_loader = DataLoader(
        test_data, 
        batch_size=len(test_data), 
        shuffle=False
    )

    shape_dataset = X_train.shape[1]

    dataset = {
        "train_input" : X_train, 
        "train_label" : y_train, 
        "test_input" : X_test, 
        "test_label" : y_test,
        "scaler_cont": scaler,
        "scaler_ord": ordinal_encoder,
        "label_encoders": label_encoders,
        "continuous_labels": continuous_labels, 
        "binary_labels": binary_labels,
        "ordinal_labels": ordinal_labels,
        "original_data": original_data, 
        "original_continuous_indices": continuous_indices, 
        "original_ordinal_indices": ordinal_indices,
        "original_binary_indices": binary_indices
    }

    return dataset