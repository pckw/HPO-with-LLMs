#import csv
#from datetime import datetime
from sklearn.model_selection import ParameterSampler
from Hyperparametertuning.src.util import train_classifier
import numpy as np
import json

class RandomSearch_Tuner:
    def __init__(self, classifier_type: str, train_data: dict, test_data: dict) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.classifier_type = classifier_type

    def tune(
            self,
            parameter_grid: dict,
            n_iter: int,
            n_epochs: int,
            fout: str
        ) -> None:
        """
        Tune the model by performing hyperparameter search using random sampling.

        Args:
            parameter_grid (dict): A dictionary containing the hyperparameter grid to search over.
            n_iter (int): The number of parameter settings that are sampled.
            n_epochs (int): The number of times to repeat the hyperparameter search.

        Returns:
            list of lists of floats: A nested list containing the losses for each combination of hyperparameters.

        """
        rng = np.random.RandomState(0)
        param_list = [
            list(
                ParameterSampler(
                    parameter_grid,
                    n_iter=n_iter,
                    random_state=rng
                )
            )
            for i in range(n_epochs)
        ]
        list_of_logs = []
        for e, param in enumerate(param_list):
            losses = [1]
            print(f"Epoch {e}")
            #losses_epoch = []
            for i, p in enumerate(param):
                log = dict()
                loss = train_classifier(
                    classifier_type=self.classifier_type,
                    train_data=self.train_data,
                    test_data=self.test_data,
                    config=p
                )
                if loss < min(losses):
                    print(f"New best loss: {loss}")
                    print(f"Config: {p}")
                    print("-------------")
                losses.append(loss)
                log['epoch'] = e
                log['iteration'] = i
                log['config'] = {i: j for i, j in zip(p.keys(), p.values())}
                #log['config'] = {i: float(j) for i, j in zip(p.keys(), p.values())}
                log['loss'] = float(loss)
                list_of_logs.append(log)
            #losses_random_search.append(losses_epoch)
        # write log to json
        with open(fout, 'w') as f:
            json.dump(list_of_logs, f, indent=4)
        return list_of_logs
