import time
import torch
import torch.nn as nn
import dill
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

from models.basic_mlp_net import BasicNet
from kan.efficient_kan import KAN as EKAN
from pykan.kan import MultKAN

import torch
import numpy as np
from tqdm import tqdm
import re

def train_pykan(
    epochs: int, 
    model,  # MultKAN model
    dataset: dict[str, torch.Tensor], 
    experiment: str = None
) -> tuple[float, dict[str, np.ndarray]]:
    print(f'\033[34mexperiment:\033[37m {experiment}')

    start_time = time.time()

    results = model.fit(dataset, opt="LBFGS", steps=epochs, loss_fn=torch.nn.CrossEntropyLoss())

    end_time = time.time()
    avg_epoch_time = (end_time - start_time) / epochs

    # Sanitize experiment name
    safe_experiment = re.sub(r'[^\w\-_.]', '_', experiment) if experiment else "trained"
    save_name = f"{safe_experiment}_multkan.pkl"

    # Save using dill
    with open(save_name, 'wb') as f:
        dill.dump(model, f)
    print(f"MultKAN model saved to '{save_name}' using dill")

    return avg_epoch_time, results
