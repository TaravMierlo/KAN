import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from pykan.kan import KAN
from train.train import train_pykan

# Constants
KAN_SHAPE = [53, 1, 2]
GRID = 5
K = 3
SEED = 42
NUM_EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load and prepare data
data = pd.read_stata("datasets/MIMIC-IV.dta")
data = data.drop([
    "deliriumtime", "hosp_mort", "icu28dmort", "stay_id", "icustay", 
    "hospstay", "sepsistime"
], axis=1).dropna()

X = data.drop(columns=["SAD"])
y = data["SAD"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED)

dataset = {
    "train_input": torch.tensor(X_train, dtype=torch.float32),
    "train_label": torch.tensor(y_train.values, dtype=torch.float32),
    "test_input": torch.tensor(X_test, dtype=torch.float32),
    "test_label": torch.tensor(y_test.values, dtype=torch.float32),
}

# Build and train model
model = KAN(
    width=KAN_SHAPE,
    grid=GRID,
    k=K,
    seed=SEED,
    device=DEVICE,
    sparse_init=True
)

print("Training model...")
train_pykan(
    epochs=NUM_EPOCHS,
    model=model,
    dataset=dataset,
    experiment="PyKAN | dataset=SAD"
)

# Prune and save model
model.prune()

torch.save({
    'model_state_dict': model.state_dict(),
    'shape': KAN_SHAPE,
    'grid': GRID,
    'k': K,
    'seed': SEED
}, 'kan_sad_full.pth')

print("Model saved to kan_sad_full.pth")
