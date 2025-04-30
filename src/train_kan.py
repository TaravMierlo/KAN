import os
import datetime
import torch
import matplotlib.pyplot as plt

from pykan.kan import KAN
from utils.utils import count_parameters, compute_metrics
from train.train import train_pykan

def train_kan_model(
    dataset,
    original_data,
    kan_shape=[53, 1, 2],
    num_epochs=10,
    seed=42,
    grid=5,
    k=3,
    with_graphs=True,
    device=None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%M')
    output_path = f"output/SAD/{timestamp}"

    if with_graphs and not os.path.exists(output_path):
        os.makedirs(f"{output_path}/train-test-loss", exist_ok=True)
        os.makedirs(f"{output_path}/network-plots", exist_ok=True)

    # Move tensors to device
    for key in dataset:
        dataset[key] = dataset[key].to(device)

    # Initialize model
    model = KAN(
        width=kan_shape, grid=grid, k=k,
        seed=seed, device=device, sparse_init=True
    )

    # Train the model
    timings, results = train_pykan(
        epochs=num_epochs,
        model=model,
        dataset=dataset,
        experiment=f"PyKAN | dataset=SAD | KAN shape={kan_shape}"
    )

    model.prune()

    if with_graphs:
        # Loss plot
        plt.clf()
        plt.plot(results['train_loss'])
        plt.plot(results['test_loss'])
        plt.legend(['train', 'test'])
        plt.title("Loss per step")
        plt.ylabel('Loss')
        plt.xlabel('Step')
        plt.yscale('log')
        plt.savefig(f"{output_path}/train-test-loss/seed={seed}_epochs={num_epochs}.png")
        plt.show()
        plt.clf()

        # Network plot
        model.plot(
            folder=f"{output_path}/network-plots/loose-plots_seed={seed}_epochs={num_epochs}/",
            tick=True,
            in_vars=list(original_data.drop("SAD", axis=1).columns),
            out_vars=list(range(0, 2)),
            varscale=0.1,
            title=f"seed={seed} epochs={num_epochs}",
            unnormalised_DataFrame=original_data
        )
        plt.savefig(f"{output_path}/network-plots/seed={seed}_epochs={num_epochs}.png", dpi=4000)
        plt.show()

    # Evaluate
    model.to('cpu')
    model.eval()
    with torch.no_grad():
        data = dataset["test_input"].to('cpu')
        target = dataset["test_label"].to('cpu')
        output = model(data)
        correct = (output.argmax(1) == target).sum().item()
        total = target.size(0)

        t_f1, test_precision, test_recall, FPR, FNR = compute_metrics(
            target=target, output=output, average='macro'
        )

    test_acc = correct / total * 100

    return {
        "model": model,
        "metrics": {
            "accuracy": test_acc,
            "f1": t_f1 * 100,
            "precision": test_precision * 100,
            "recall": test_recall * 100,
            "fpr": FPR * 100,
            "fnr": FNR * 100,
        },
        "training_time_per_epoch": timings,
        "parameters": count_parameters(model),
        "timestamp": timestamp,
        "output_path": output_path,
    }
