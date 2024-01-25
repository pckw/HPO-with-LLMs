import numpy as np
import xgboost
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import json


def subsample(train: dict, test: dict, f: float) -> tuple[dict, dict]:
    """
    Subsamples the train and test datasets based on a given factor.

    Args:
        train (dict): The training dataset.
        test (dict): The testing dataset.
        f (float): The subsampling factor.

    Returns:
        tuple[dict, dict]: A tuple containing the subsampled training and testing datasets.
    """
    train_embedding = []
    train_target = []
    test_embedding = []
    test_target = []
    for k in range(len(train['target_names'])):
        ii_train =int(sum(train['target'] == k) * f)
        ii_test =int(sum(test['target'] == k) * f)
        train_embedding.extend([i for i,j in zip(train['embedding'], train['target']) if j == k][0:ii_train])
        train_target.extend([i for i in train['target'] if i == k][0:ii_train])
        test_embedding.extend([i for i,j in zip(test['embedding'], test['target']) if j == k][0:ii_test])
        test_target.extend([i for i in test['target'] if i == k][0:ii_test])
    train['embedding'] = np.array(train_embedding)
    train['target'] = np.array(train_target)
    test['embedding'] = np.array(test_embedding)
    test['target'] = np.array(test_target)
    return train, test


def train_classifier(
        classifier_type: str,
        train_data: dict,
        test_data: dict,
        config: dict = None
) -> float:
        """
        Trains a classifier using the provided training data and evaluates its performance on the test data.

        Args:
            train_data (dict): A dictionary containing the training data, with 'embedding' and 'target' keys.
            test_data (dict): A dictionary containing the test data, with 'embedding' and 'target' keys.
            config (dict, optional): A dictionary containing the configuration parameters for the classifier. 
                                    Defaults to None.

        Returns:
            float: The classification error rate, calculated as 1 minus the accuracy score of the classifier on the 
                   test data.
        """
        if classifier_type == 'xgboost':
            if config:
                classifier = xgboost.XGBClassifier(**config)
            else:
                classifier = xgboost.XGBClassifier()
        if classifier_type == 'svc':
            if config:
                classifier = SVC(**config)
            else:
                classifier = SVC()
        classifier.fit(train_data['embedding'], train_data['target'])
        return 1-classifier.score(test_data['embedding'], test_data['target'])


def plot_losses(
        file_rs: str,
        file_llm: str,
        ylim1: tuple,
        ylim2: tuple,
        file_automl: str = None,
        print_param: bool = False,
        fout: str = None
):
    """
    Plots the losses from the given files and optional automl file.

    Args:
        file_rs (str): Path to the random search file.
        file_llm (str): Path to the llmtest file.
        ylim1 (tuple): Y-axis limits for the first subplot.
        ylim2 (tuple): Y-axis limits for the second subplot.
        file_automl (str, optional): Path to the automl file (default is None).
        print_param (bool, optional): Whether to print kernel and C parameters (default is False).
        fout (str, optional): Output file path to save the plot (default is None).
    """
    # read loss from llmtest.json
    with open(file_llm, "r") as f:
        data_llm = json.load(f)
    if print_param:
        kernel_llm = [i['config']['kernel'] for i in data_llm]    
        c_llm = [i['config']['C'] for i in data_llm]
    loss_llm = [i['loss'] for i in data_llm]

    # read rows from csv into list of lists
    with open(file_rs, "r") as f:
        data_rs = json.load(f)
    n_epoch = max([i['epoch'] for i in data_rs])
    loss_rs = [min([i['loss'] for i in data_rs if i['epoch'] == n]) for n in range(n_epoch+1)]  
    # automl loss
    if file_automl:
        with open(file_automl, "r") as f:
            logfile = f.read().split("\n")
        logfile = [json.loads(line) for line in logfile if line != ""]
        loss_automl = [f['validation_loss'] for f in logfile[:-1] if f['record_id'] == logfile[-1]['curr_best_record_id']][0]

    llm_beats_rs = [min(loss_llm) < i for i in loss_rs]
    print(f"LLM beats Random Search: {sum(llm_beats_rs)} / {len(llm_beats_rs)}")
    if file_automl:
        automl_beats_rs = [loss_automl < i for i in loss_rs]
        print(f"AutoML beats Random Search: {sum(automl_beats_rs)} / {len(automl_beats_rs)}")

    fig, ax = plt.subplots(2,1, figsize=(10,12))
    ax[0].plot(loss_llm, label="GPT4", color="k", lw=3)

    # bar plot of loss rs
    ax[1].bar(np.arange(len(loss_rs)), loss_rs, label="Random search", color="C0")
    if file_automl:
        ax[1].axhline(y=loss_automl, color='gray', label="AutoML", lw=3)
    ax[1].axhline(y=min(loss_llm), color='k', label="GPT-4-turbo", lw=3)
    if len(loss_llm) > 10:
        ax[1].axhline(y=min(loss_llm[:10]), color='k', ls=':', label="GPT-4-turbo (10 Iterations)", lw=3)
    ax[0].set_title("a) GPT-4-turbo", size=16)
    ax[1].set_title("b) GPT-4-turbo vs. Random Search", size=16)
    ax[1].legend(fontsize=14)
    ax[0].set_xlabel("Iteration", size=14)
    ax[1].set_xlabel("Epoch", size=14)
    for a in ax:
        a.set_ylabel("Validation Error Rate", size=14)
        a.tick_params(axis='both', which='major', labelsize=14)
    ax[0].set_xticklabels(np.arange(1, len(loss_llm)+1), size=14)
    ax[1].set_xticklabels(np.arange(1, len(loss_rs)+1), size=14)
    ax[0].set_ylim(ylim1)
    ax[1].set_ylim(ylim2)
    ax[0].set_xticks(np.arange(len(loss_llm)));
    ax[1].set_xticks(np.arange(len(loss_rs)));
    if print_param:
        for i, k in enumerate(zip(kernel_llm, c_llm)):
            ax[0].text(i,ylim1[1]*0.84,k[0], ha='center', rotation=45, size=14)
            ax[0].text(i,ylim1[1]*0.76,np.round(k[1],3), ha='center', rotation=45, size=14)
    if fout:
        fig.savefig(fout)
