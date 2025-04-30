import statistics 
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Set the random seeds for reproducibility pytorch
def seed_everything(seed=0):
    import random
    random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(target, output, average='macro'):
    
    
    # Calculate f1 score
    f1 = f1_score(target.cpu().numpy(), output.argmax(1).cpu().numpy(), average=average, zero_division=0.0)
    # Calculate precision score
    precision = precision_score(target.cpu().numpy(), output.argmax(1).cpu().numpy(), average=average, zero_division=0.0)
    # Calculate recall score
    recall = recall_score(target.cpu().numpy(), output.argmax(1).cpu().numpy(), average=average, zero_division=0.0)

    if average == 'macro':
        # Obtain the confusion matrix
        cm = confusion_matrix(target, output.argmax(1))
        TN, FP, FN, TP = cm.ravel()

        # Calculate FPR and FNR
        FPR = FP / (FP + TN)
        FNR = FN / (FN + TP)

        return f1, precision, recall, FPR, FNR
    else:
        return f1, precision, recall, 0, 0

def write_results(f, nodes, parameters, papi_results, test_accuracies, test_f1, test_precision_scores, test_recall_scores, test_fpr, test_fnr, times):
    
    # Write all on the result file
    f.write(f"Nodes:{nodes}\n")
    f.write(f"Parameters:{parameters}\n")
    
    f.write(f"PAPI Results:{papi_results}\n")
   
    f.write(f"Test Accuracy\n")
    for el in test_accuracies:
      
        f.write(f"{el}\n")
    f.write(f"Test F1\n")
    for i in test_f1:    
        f.write(f"{i}\n")
    f.write(f"Test Precision\n")
    for i in test_precision_scores:
        f.write(f"{i}\n")
    f.write(f"Test Recall\n")
    for i in test_recall_scores:
        f.write(f"{i}\n")
    f.write(f"Test FPR\n")
    for i in test_fpr:
        f.write(f"{i}\n")
    f.write(f"Test FNR\n")
    for i in test_fnr:
        f.write(f"{i}\n")
    f.write(f"     Training time(s)\n")
    for i in times:
        f.write(f"{i}\n")
    f.write("_____________________\n")

def nice_printout(f)-> None:
    """
    It is probably not worth figuring out how this parser works...
    I did not bother writing proper documentation for it.
    """

    def parse_data(txt: list[str])-> tuple[str, float, int]:
        average = 0
        i = 1
        while "T" not in txt[i] and "_" not in txt[i]:
            average += float(txt[i])
            i += 1
        return txt[0][5:], average / max(1, i-1), max(1, i-1)

    def parse_seed(txt: list[str])-> tuple[dict[str, float], int]:
        averages = {}
        i = 0
        while "_" not in txt[i]:
            if "T" in txt[i]:
                column, average, new_i = parse_data(txt[i:])
                i += new_i
                averages[column[:-1]] = average
            else: 
                i += 1
        return averages, i

    def seed_dicts_to_str(seed_dicts: list[dict[str, float]])-> str:
        """
        It is to be assumed that all dicts have the same keys. 
        If thats not the case, tough shit.
        """
        output = ""
        for key in seed_dicts[0].keys():
            data = [seed_dict[key] for seed_dict in seed_dicts]
            if key == "Training time(s)":
                output += f"│ {key[:-3]:<15}│ " \
                f"{(statistics.mean(data)):<9.2f}secs "
            else:
                output += f"│\033[4m {key:<15}│ {max(data):<6.2f} │ "
                if len(data) > 1:
                    output += f"{statistics.stdev(data):<5.2f}"
                else:
                    output += f"{'nan':<5}"
            output += f" \033[0m│\n"
        return output

    filename = f.name
    f.close()
    with open(filename, "r") as file:
        txt = file.readlines()

    model_type = "PyKAN" if "pykan" in filename else (
        "MLP" if "mlp" in filename else "kan"
    )

    dataset_name = txt[0].split(': ')[-1].replace('_', ' ') \
        .replace(' pykan', '').title()[:-1]
    output = f"{'Dataset':<8}: {dataset_name}\n" \
        f"{'Model':<8}: {model_type}\n┌{'─'*16}┬{'─'*8}┬{'─'*7}┐\n" \
        f"│\033[4m {'metric':^15}│ {'%':^6} │ {'std':^5} \033[0m│\n"
    i = 0
    seed_outputs = []
    while i < len(txt):
        if "Seed: " in txt[i]:
            seed_output, new_i = parse_seed(txt[i+1:])
            i += new_i
            seed_outputs.append(seed_output)
        else:
            i += 1 
    output += seed_dicts_to_str(seed_outputs) + f"└{'─'*16}┴{'─'*16}┘" 

    print(output)
    print(
        "\033[31mNOTE: This table contains the maximum scores "
        "(and their standard deviations) for each of the seeds.\nWithin "
        "the seed everything gets averaged. Time is just the average.\033[37m"
    )
