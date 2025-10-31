# %%
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mlp_utils import DSArray, trainMLP
from sklearn.preprocessing import LabelEncoder
import lightning as L
import argparse
import optuna

# %%
# set seed
L.seed_everything(42)

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--llm', type=str, default='Llama-4-Scout-17B-16E')
parser.add_argument('--embp', type=str, default='embSumRaw')
args = parser.parse_args()

llm = args.llm
embp = args.embp # embSumRaw or embSumT1

# %%
# load embeds
X = torch.load(f'/rcp/nallapar/cshl/data/{llm}_{embp}.pt').numpy()
ds = pd.read_csv('/rcp/nallapar/cshl/data/encode_experiments_summary.csv')

# %%
tasks = ["lab", "assay_title", "cell_type", "classification", "num_technical_reps", "num_biological_reps", "assay_term_name", "ended", "mrl", "read_length", "replication_type", "read_depth", "assembly", "status", "spot_score", "frip_score", "species", "fold change over control", "signal p-value", "peaks", "targets"]

def objective(trial):
    # define hyperparameters to optimize
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    bs = trial.suggest_categorical('bs', [16, 32, 64, 128])
    dropout_val = trial.suggest_float('dropout_val', 0.1, 0.3)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    num_nodes = trial.suggest_categorical('num_nodes', [32, 64, 128, 256, 512])
    max_epochs = 200

    f1_dict = {} 
    balacc_dict = {}

    # %%
    for t in tasks:
        # drop rows with NA string in the task column
        ds_dropna = ds[ds[t] != 'NA']
        # remove nan values
        ds_dropna = ds_dropna.dropna(subset=[t])
        
        y = list(ds_dropna[t])
        le = LabelEncoder()
        y = le.fit_transform(y)

        # remove samples whose classes only have a frequency of 1
        val, count = np.unique(y, return_counts=True)
        singleton_classes = []
        for i in range(len(count)):
            if count[i] < 3:
                singleton_classes.append(val[i])

        indices_to_remove = []
        for i in range(len(y)):
            if y[i] in singleton_classes or y[i] == 'NA' or y[i] is None or np.isnan(y[i]):
                indices_to_remove.append(i)

        indices_to_keep = [i for i in range(len(y)) if i not in indices_to_remove]

        X_proc = X[indices_to_keep]
        y_proc = y[indices_to_keep]

        X_train, X_test, y_train, y_test = train_test_split(X_proc, y_proc, test_size=0.2, random_state=42, stratify=y_proc)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

        train_dataset = DSArray(X_train, y_train)
        val_dataset = DSArray(X_val, y_val)
        test_dataset = DSArray(X_test, y_test)

        # convert to dataloaders
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

        print("Task: " + t)
        print("Number of classes: ", len(list(set(y))))
        print("Number of samples: ", len(y_proc), len(X_proc))

        # logreg model for assay title
        save_loc = 'saved_models/MLP_' + str(llm) + '_' + str(embp) + '_' + str(lr) + '_' + str(bs) + '_' + str(dropout_val) + '_' + str(num_layers) + '_' + str(num_nodes) + '_' + str(t) + '/'
        out_model, results = trainMLP(train_dataset.inp_ft, len(list(set(y))), max_epochs, bs, lr, save_loc, train_dataloader, test_dataloader, val_dataloader, dropout_val, num_layers, num_nodes)

        f1_dict[t] = results[0]['test/f1']
        balacc_dict[t] = results[0]['test/balacc']

    # %%
    # all performances
    print("### F1 Scores ###\n")
    print(f1_dict)
    print("\n### Bal Acc ###\n")
    print(balacc_dict)

    # %%
    # grouped metrics
    # experiment quality: replication_type, read_depth, spot_score, frip_score, mrl, read_length
    f1_exp_quality = (f1_dict['replication_type'] + f1_dict['read_depth'] + f1_dict['spot_score'] + f1_dict['frip_score'] + f1_dict['mrl'] + f1_dict['read_length']) / 6
    balacc_exp_quality = (balacc_dict['replication_type'] + balacc_dict['read_depth'] + balacc_dict['spot_score'] + balacc_dict['frip_score'] + balacc_dict['mrl'] + balacc_dict['read_length']) / 6
    print("Mean F1 - Experiment Quality: ", f1_exp_quality)
    print("Mean BalAcc - Experiment Quality: ", balacc_exp_quality)

    # biosample information: classification, cell_type, species
    f1_biosample_info = (f1_dict['classification'] + f1_dict['cell_type'] + f1_dict['species']) / 3
    balacc_biosample_info = (balacc_dict['classification'] + balacc_dict['cell_type'] + balacc_dict['species']) / 3
    print("Mean F1 - Biosample Info: ", f1_biosample_info)
    print("Mean BalAcc - Biosample Info: ", balacc_biosample_info)

    # experiment protocol: assay_title, num_technical_reps, num_biological_reps, assay_term_name, ended, assembly, targets, lab
    f1_exp_protocol = (f1_dict['assay_title'] + f1_dict['num_technical_reps'] + f1_dict['num_biological_reps'] + f1_dict['assay_term_name'] + f1_dict['ended'] + f1_dict['assembly'] + f1_dict['targets'] + f1_dict['lab']) / 8
    balacc_exp_protocol = (balacc_dict['assay_title'] + balacc_dict['num_technical_reps'] + balacc_dict['num_biological_reps'] + balacc_dict['assay_term_name'] + balacc_dict['ended'] + balacc_dict['assembly'] + balacc_dict['targets'] + balacc_dict['lab']) / 8
    print("Mean F1 - Experiment Protocol: ", f1_exp_protocol)
    print("Mean BalAcc - Experiment Protocol: ", balacc_exp_protocol)

    # output categories: fold change over control, signal p-value, peaks
    f1_output_cat = (f1_dict['fold change over control'] + f1_dict['signal p-value'] + f1_dict['peaks']) / 3
    balacc_output_cat = (balacc_dict['fold change over control'] + balacc_dict['signal p-value'] + balacc_dict['peaks']) / 3
    print("Mean F1 - Output Cat: ", f1_output_cat)
    print("Mean BalAcc - Output Cat: ", balacc_output_cat)

    f1_overall_except_outcat = (f1_exp_quality*6 + f1_biosample_info*3 + f1_exp_protocol*8) / 17
    balacc_overall_except_outcat = (balacc_exp_quality*6 + balacc_biosample_info*3 + balacc_exp_protocol*8) / 17
    print("Mean F1 - Overall except Output Cat: ", f1_overall_except_outcat)
    print("Mean BalAcc - Overall except Output Cat: ", balacc_overall_except_outcat)

    return 1-f1_overall_except_outcat

# %%
# define optuna study
study = optuna.create_study()
# optimize lr, bs, dropout_val, num_layers, num_nodes

study.optimize(objective, n_trials=100)
# print best trial
print("\nBest trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))