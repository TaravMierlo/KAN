import torch
import torch.utils
import numpy as np
import datetime

from data.loading_data import prepare_dataset
from train.train import train, train_pykan
from models.basic_mlp_net import BasicNet
from kan.efficient_kan import KAN as EKAN
from utils.utils import seed_everything, count_parameters, compute_metrics, write_results, nice_printout
from pykan.kan import *


# The names of the target columns in the datasets, used for plotting.
TARGETS_PER_DATASET = {
    "SAD" : "SAD",
    "breast_cancer" : "Class",
}

def run_single_model(
    f: str, 
    dataset_name: str, 
    model_name: str, 
    shape_dataset: int, 
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_epochs: int, 
    num_classes: int, 
    device: str, 
    num_models: int,
    average: str='macro'
)-> None:
    """
    Train a single model, kan/mlp version.

    This function takes in a dataset and some settings and trains
     a subset of models.

    @param f (str): output file to write results to
    @param dataset_name (str): can be any of [
        'SAD', 
        'breast_cancer', 
        'spam', 
        'musk', 
        'dry_bean', 
        'gamma', 
        'adult', 
        'shuttle', 
        'diabetes', 
        'poker'
    ]
    @param model_name (str): Can be one of ['kan', 'mlp'].
     All other names will fail for this function.
    @param shape_dataset (int): Number of input features.
    @param train_loader (torch.utils.data.DataLoader): Training data.
    @param test_loader (torch.utils.data.DataLoader): testning data.
    @param num_epochs (int): number of training 'steps' for each model. 
    @param num_classes (int): number of output features/classes.
    @param device (str): Either 'cuda' or 'cpu', to indicate device to
     train the models on.
    @param num_models = The number of models ran. 
     KAN models start with 1 and increase by 1. 
     MLP models start with 10 and increase by 10.
    @param average (str) [DEFAULT='macro']: passed to `compute_metrics`.
    """
    
    nodes = []
    training_time_per_epoch = []
    parameters = []
    papi_results = []
    test_accuracies = []
    test_f1 = []
    test_precision_scores = []
    test_recall_scores = []
    test_fpr = []
    test_fnr = []

    if model_name.lower() not in ['kan', 'mlp']:
        raise NotImplementedError(
            f"This function does currently support {model_name}."
        )

    # Set the number of intermediate nodes
    # change this range to train less models per seed, 
    #  or to alter the number of nodes per model.
    if model_name == 'kan':
        num_intermediate_nodes = np.arange(1, (1 + num_models), 1)
    else:
        # give mlp ranges 10-100 (steps of 10)
        num_intermediate_nodes = np.arange(10, (10 + num_models * 10), 10)

    # Because of the difficulty of learning the poker set, 
    # it gets 10x as many nodes.
    if dataset_name == 'poker' or dataset_name == 'poker_pykan':
        num_intermediate_nodes*=10

    for n in num_intermediate_nodes:

        if model_name == 'kan':
            # EKAN is the EfficientKAN imlementation
            model = EKAN([shape_dataset, n, num_classes])
        else:
            # BasicNet is the MLP
            model = BasicNet(shape_dataset, num_classes, n)
    
        # Send model to correct device
        model.to(device)

        # Train model and save timings
        training_time_per_epoch.append(
            train(
                epochs=num_epochs, 
                model=model, 
                device=device, 
                train_loader=train_loader, 
                experiment=f"\033[32m{model_name} (n_nodes={n}) |"
                    f" dataset={dataset_name}\033[37m"
            )
        )
        
        #################################
        #       Calculate metrics       #
        #################################

        # Count and save how many nodes were used.
        nodes.append(n)

        # Count and save how many parameters that is.
        parameters.append(count_parameters(model))

        # We need to change the device to cpu for the sake of efficiency.
        device = 'cpu'
        model.to(device)
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                
                total += target.size(0)
                
                # Use torch.argmax to find the predicted class.
                correct += (output.argmax(1) == target).sum().item()

                t_f1, test_precision, test_recall, FPR, FNR = compute_metrics(
                    target=target, 
                    output=output, 
                    average=average
                )
                
                # Convert all to percentages before saving.
                test_f1.append(t_f1*100)
                test_precision_scores.append(test_precision*100)
                test_recall_scores.append(test_recall*100)
                test_fpr.append(FPR*100)
                test_fnr.append(FNR*100)

        test_acc = correct / total * 100
        test_accuracies.append(test_acc)

    # Write all results to file
    write_results(
        f, 
        nodes, 
        parameters, 
        papi_results, 
        test_accuracies, 
        test_f1, 
        test_precision_scores, 
        test_recall_scores, 
        test_fpr, 
        test_fnr, 
        training_time_per_epoch
    )

def run_single_model_pykan(
    f: str, 
    dataset_name: str, 
    model_name: str, 
    shape_dataset: int, 
    dataset: dict[str : torch.Tensor], 
    seed: int, 
    num_epochs: int, 
    num_classes: int, 
    with_graphs: bool, 
    original_data: pd.DataFrame, 
    device: str, 
    num_models: int,
    average: str='macro'
)-> None:
    """
    Train a single model, Pykan version.

    This function takes in a dataset and some settings and trains
     a subset of models.

    @param f (str): output file to write results to
    @param dataset_name (str): can be any of [
        'SAD', 
        'breast_cancer', 
        'spam', 
        'musk', 
        'dry_bean', 
        'gamma', 
        'adult', 
        'shuttle', 
        'diabetes', 
        'poker'
    ]
    @param model_name (str): Can be one of ['pykan'].
     All other names will fail for this function.
    @param shape_dataset (int): Number of input features.
    @param dataset (dict[str : torch.Tensor]): Dataset with keys [
        "train_input",
        "train_label",
        "test_input",
        "test_label"
    ]
    @param seed (int): Seed for initializing model and saving plots.
    @param num_epochs (int): number of training 'steps' for each model. 
    @param num_classes (int): number of output features/classes.
    @param with_graphs (bool): True if you want to output the KAN graphs
     for each model trained.
    NOTE: This has a significant impact on the runtime, 
     as the images are huge.
    @param original_data (pd.DataFrame): The original (untouched) 
     DataFrame. Only used if with_graphs is True.
    @param device (str): Either 'cuda' or 'cpu', to indicate device to
     train the models on.
    @param num_models (int): The number of models ran. 
     (Py)KAN models start with 1 and increase by 1. 
    @param average (str) [DEFAULT='macro']: passed to `compute_metrics`.
    """
    
    #create output folder for the plots etc.
    if with_graphs:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%Hh%M')
        if (not os.path.exists(f"output/{dataset_name}/{timestamp}")):
            os.makedirs(f"output/{dataset_name}/{timestamp}")
            os.mkdir(f"output/{dataset_name}/{timestamp}/train-test-loss")
            os.mkdir(f"output/{dataset_name}/{timestamp}/network-plots")

    nodes = []
    training_time_per_epoch = []
    parameters = []
    papi_results = []
    test_accuracies = []
    test_f1 = []
    test_precision_scores = []
    test_recall_scores = []
    test_fpr = []
    test_fnr = []

    # send training data to device
    for key, value in dataset.items():
        dataset[key] = value.to(device)
  
    if model_name != 'pykan':
        raise NotImplementedError(
            "You cant do anything but kans for the pykan variants."
        )
    
    # Set the number of intermediate nodes
    # change this range to train less models per seed, 
    #  or to alter the number of nodes per model.
    num_intermediate_nodes = np.arange(1, (1 + num_models), 1)

    # Because of the difficulty of learning the poker set, 
    # it gets 10x as many nodes.
    if dataset_name == 'poker' or dataset_name == 'poker_pykan':
        num_intermediate_nodes*=10

    # For each stated configuration, train a model.
    for n in num_intermediate_nodes:
        # Create KAN model with shape [
        #     number of input features,
        #     number of nodes decided by `num_intermediate_nodes`
        #     number of output classes
        # ]
        model = KAN(
            width=[int(shape_dataset), int(n), int(num_classes)], 
            grid=5, 
            k=3, 
            seed=seed, 
            device=device
        )
        
        # Train the pykan model
        timings, results = train_pykan(
            epochs=num_epochs, 
            model=model, 
            dataset=dataset,
            experiment=f"\033[32m{model_name} (n_nodes={n}) | "
                f"dataset={dataset_name} | KAN shape="
                f"{[int(shape_dataset), int(n), int(num_classes)]}\033[37m"
        )

        #################################
        #          save graphs          #
        #################################

        if with_graphs:
            # plot the loss over time.
            plt.clf()
            plt.plot(results['train_loss'])
            plt.plot(results['test_loss'])
            plt.legend(['train', 'test'])
            plt.title("loss per step")
            plt.ylabel('loss')
            plt.xlabel('step')
            plt.yscale('log')
            plt.savefig(
                f"output/{dataset_name}/{timestamp}/train-test-loss/"
                f"seed={seed}_nodes={n}_epochs={num_epochs}.png"
            )

            plt.clf()

            # Depending on the dataset, the target might not be known. 
            # This can be set in `TARGETS_PER_DATASET`.
            # If not present, no labels will be put on the Pykan axes.
            try:
                # Get all the columns as a list of strings, 
                #  without the target column.
                in_vars = list(
                    original_data.drop(
                        TARGETS_PER_DATASET[dataset_name], 
                        axis=1
                    ).columns
                )
            except:
                in_vars = None

            # Plot the Pykan model.
            model.plot(
                folder=f"output/{dataset_name}/{timestamp}/network-plots/"
                    f"loose-plots_seed={seed}_nodes={n}_epochs={num_epochs}/", 
                tick=True,
                in_vars=in_vars,
                out_vars=list(range(0, num_classes)),
                varscale=0.1,
                title=f"seed={seed} nodes={n} epochs={num_epochs}"
            )
            # Save the plot with a very high dpi (dots per inch).
            # This makes really big output images, 
            #  which keeps you able to zoom in.
            plt.savefig(
                f"output/{dataset_name}/{timestamp}/network-plots/"
                    f"seed={seed}_nodes={n}_epochs={num_epochs}.png", 
                dpi=4000
            )

        #################################
        #       Calculate metrics       #
        #################################

        # timings defined in `train_pykan`
        training_time_per_epoch.append(timings)

        # Count and save how many nodes were used.
        nodes.append(n)

        # Count and save how many parameters that is.
        parameters.append(count_parameters(model))

        # We need to change the device to cpu for the sake of efficiency.
        test_device = 'cpu'
        model.to(test_device)
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            data = dataset["test_input"].to(test_device)
            target = dataset["test_label"].to(test_device)

            output = model(data)
            
            total += target.size(0)
            
            # Use torch.argmax to find the predicted class.
            correct += (output.argmax(1) == target).sum().item()

            t_f1, test_precision, test_recall, FPR, FNR = compute_metrics(
                target=target, 
                output=output, 
                average=average)
            
            # Convert all to percentages before saving.
            test_f1.append(t_f1*100)
            test_precision_scores.append(test_precision*100)
            test_recall_scores.append(test_recall*100)
            test_fpr.append(FPR*100)
            test_fnr.append(FNR*100)

        test_acc = correct / total * 100
        test_accuracies.append(test_acc)

    # Write all to the file
    write_results(
        f, 
        nodes, 
        parameters, 
        papi_results, 
        test_accuracies, 
        test_f1, 
        test_precision_scores, 
        test_recall_scores, 
        test_fpr, 
        test_fnr, 
        training_time_per_epoch
    )

def run_experiments(
    dataset_name: str, 
    model_name: str, 
    num_epochs: int, 
    num_seeds: int,
    num_models: int,
    with_graphs: bool
)-> None:
    """
    Run experiments

    This function defines a set of seeds, for which it then runs each of 
     the models in accordance with the provided arguments.

    @param f (str): output file to write results to
    @param dataset_name (str): can be any of [
        'SAD', 
        'breast_cancer', 
        'spam', 
        'musk', 
        'dry_bean', 
        'gamma', 
        'adult', 
        'shuttle', 
        'diabetes', 
        'poker'
    ]
    @param model_name (str): Can be one of ['pykan'].
     All other names will fail for this function.
    @param num_epochs (int): number of training 'steps' for each model.
    @param num_seeds (int): The number of times the entire setup gets 
     ran (each with different seed). 
    @param num_models (int): The number of models ran. 
     (Py)KAN models start with 1 and increase by 1. 
     MLP models start with 10 and increase by 10.
    @param with_graphs (bool): True if you want to output the KAN graphs
     for each model trained.
    NOTE: This has a significant impact on the runtime, 
     as the images are huge.
    """

    # number of classes per dataset
    classes = {
        'breast_cancer' : 2, 
        'spam' : 2, 
        'musk' : 2,         
        'dry_bean' : 7, 
        'gamma' : 2, 
        'adult' : 2, 
        'shuttle' : 7, 
        'diabetes' : 2, 
        'poker' : 10, 
        'SAD' : 2, # new
    }

    # Some of the datasets need to use 
    #  the weighted average instead of macro.
    multivariate_datasets = ['poker', 'shuttle', 'dry_bean']
    average = [
        'weighted' if dataset_name in multivariate_datasets else 'macro'
    ]

    # debug overview
    print("loading dataset...")

    X_train, X_test, y_train, y_test, \
    train_loader, test_loader, shape_dataset, \
    original_data = prepare_dataset(dataset_name)   

    seeds = list(range(0,num_seeds))

    # This should pick your CPU if you dont have a GPU. 
    # For some datasets training might be faster on CPU.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Uncomment the line below to force training on CPU:
    # device = 'cpu'

    # Let the user know which device is being used.
    device_color = '\033[32m' if device == 'cuda' else '\033[31m'
    print(f"\033[33mUsing device {device_color}{device}\033[37m")

    # Open results file
    try:
        f = open(
            f"output/{device}_results_{model_name}_{dataset_name}.txt", 
            "w"
        )
    except FileNotFoundError:
        try:
            f = open(
                f"benchmarking-KAN/output/{device}_results_"
                    f"{model_name}_{dataset_name}.txt", 
                "w"
            )
        except Exception as e:
                raise e
    # Write name of dataset
    f.write(f"Dataset: {dataset_name}\n")
    f.write("_____________________\n")

    c = classes[dataset_name]

    for s in seeds:
        seed_everything(s)
        f.write(f"Seed: {s}\n")

        # debug overview
        print(f"\033[35m\nRunning {model_name} with seed {s}.\n\033[37m")

        if model_name == "pykan":
            # Pykan works a little differently than your standard MLP 
            #  and or EfficientKAN. Hence we need to structure the 
            #  dataset a little differently
            dataset = {
                "train_input" : X_train, 
                "train_label" : y_train, 
                "test_input" : X_test, 
                "test_label" : y_test
            }

            run_single_model_pykan(
                f=f, 
                dataset_name=dataset_name, 
                shape_dataset=shape_dataset, 
                dataset=dataset, 
                seed=s,
                num_epochs=num_epochs, 
                num_classes=c, 
                average=average[0], 
                model_name=model_name, 
                with_graphs=with_graphs, 
                original_data=original_data, 
                num_models=num_models,
                device=device
            )
        elif model_name == "kan" or model_name == 'mlp':
            run_single_model(
                f=f, 
                dataset_name=dataset_name, 
                shape_dataset=shape_dataset, 
                train_loader=train_loader, 
                test_loader=test_loader,
                num_epochs=num_epochs, 
                num_classes=c, 
                average=average[0], 
                model_name=model_name, 
                num_models=num_models,
                device=device
            )
        else:
            raise ValueError(
                "Provided `model_name` is invalid. You provided: '"
                f"{model_name}', choices: ['kan', 'mlp', 'pykan']"
            )
        
    nice_printout(f)
