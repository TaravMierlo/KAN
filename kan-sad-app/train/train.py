import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

from models.basic_mlp_net import BasicNet
from kan.efficient_kan import KAN as EKAN
from pykan.kan import MultKAN


def train(
    epochs: int, 
    model: EKAN | BasicNet, 
    device: str, 
    train_loader: torch.utils.data.DataLoader, 
    experiment: str=None
)-> float:
    """
    Train function for normal kan and mlp models.

    This function will train the provided model for the given number of
     epochs. It will do so using the provided device and training data.

    @param epochs (int): number of training 'steps' for each model. 
    @param model (EKAN | BasicNet): model to train.
    @param device (str): Either 'cuda' or 'cpu', to indicate device to
     train the models on.
    @param train_loader (torch.utils.data.DataLoader): Training data.
    @param experiment (str): Prefix for the tqdm progress bar

    @returns float with runtime in seconds
    """
    optimizer= AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = ExponentialLR(optimizer, gamma=0.8)
    loss = nn.CrossEntropyLoss()

    # Times of training epochs
    times = []

    for epoch in tqdm(range(epochs), desc = f'\033[34mexperiment:\033[37m {experiment}'):
        start_time = time.time()
        model.train()
        closs = 0
        pred = []
        gt = []
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)            
            loss_value = loss(output, target.long())
            loss_value.backward()
            optimizer.step()
            closs += loss_value.item()
            pred.append(output.argmax(1))
            gt.append(target)
    
        pred = torch.cat(pred).float()
        gt = torch.cat(gt).float()
        accuracy_train = (pred == gt).float().mean().item()
        # Calculate f1 score
        f1 = f1_score(gt.cpu(), pred.cpu(), average='macro')

        scheduler.step()
        end_time = time.time()

        times.append(end_time-start_time)

    # Return the average times of training epochs
    t = sum(times)/len(times)
    return t

def train_pykan(
    epochs: int, 
    model: MultKAN, 
    dataset: dict[str, torch.Tensor], 
    experiment: str=None
)-> tuple[float, dict[str, np.ndarray]]:
    """
    Train function for normal kan and mlp models.

    This function will train the provided model for the given number of
    epochs. It will do so using the provided device and training data.

    @param epochs (int): number of training 'steps' for each model. 
    @param model (MultKAN): model to train.
    @param dataset (dict[str, torch.Tensor]): Dataset with keys [
        "train_input",
        "train_label",
        "test_input",
        "test_label"
    ]
    @param experiment (str): Prefix for the tqdm progress bar
    @returns float with runtime in seconds
    @returns dict[str, np.ndarray] with metrict results of model as dict
    """
    print(f'\033[34mexperiment:\033[37m {experiment}')
    
    start_time = time.time()
    
    results = model.fit(dataset, opt="LBFGS", steps=epochs, loss_fn=torch.nn.CrossEntropyLoss())
    
    end_time = time.time()

    return (end_time-start_time)/epochs, results
