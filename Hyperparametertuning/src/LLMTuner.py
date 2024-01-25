
import json
import openai
from dotenv import load_dotenv
import yaml
import os
import copy
from typing import cast
import re
from Hyperparametertuning.src.util import train_classifier
from datetime import datetime
import csv


class LLM_Hyperparamter_tuner():
    def __init__(self, classifier_type: str, train: dict, test: dict) -> None:
        load_dotenv(override=True)
        with open("config/config.yaml", "r", encoding="UTF-8") as config_file:
            config = yaml.safe_load(config_file)
            openai.api_base = str(config["deployment_endpoint"])
            openai.api_type = str(config["deployment_type"])
            self.model_name = str(config["chat_llm_deployment_name"])
            self.temperature = float(config["temperature"])
        openai.api_key = os.environ["APIKEY"]
        openai.api_version = "2023-03-15-preview"
        self.llm = openai.ChatCompletion()
        self.train = train
        self.test = test
        self.configs = []
        self.losses = []
        self.classifier_type = classifier_type

    def get_initial_config(self) -> tuple[dict, list]:
        """
        Retrieves the initial configuration for tuning hyperparameters of a XGBoost model.

        Parameters:
            None

        Returns:
            A tuple containing:
            - config (dict): The initial configuration in JSON format, enclosed in triple backticks.
                             Example: {"C": x, "gamma": y}
            - messages (list): A list of messages exchanged between the user and the assistant.

        Raises:
            JSONDecodeError: If the initial configuration cannot be parsed as JSON.

        Note:
            - The hyperparameter search space includes the following parameters:
                - n_estimators (Optional[int]): Number of boosting rounds.
                - max_leaves (int): Maximum number of leaves; 0 indicates no limit.
                - min_child_weight (Optional[float]): Minimum sum of instance weight (hessian) needed in a child.
                - learning_rate (Optional[float]): Boosting learning rate (xgb's "eta").
                - subsample (Optional[float]): Subsample ratio of the training instance.
                - colsample_bytree (Optional[float]): Subsample ratio of columns when constructing each tree.
                - colsample_bylevel (Optional[float]): Subsample ratio of columns for each level.
                - reg_alpha (Optional[float]): L1 regularization term on weights (xgb's alpha).
                - reg_lambda (Optional[float]): L2 regularization term on weights (xgb's lambda).

            - The budget allows trying 10 configurations in total.

            - The goal is to minimize the error rate by exploring different parts of the search space.

            - The assistant response will be included in the `messages` list.

        Example:
            config, messages = get_initial_config()
        """
        if self.classifier_type == "xgboost":
            content = """You are helping tune hyperparameters for a XGBoost model. Training is done with the XGBoost library in python. This is our hyperparameter search space:
            n_estimators (Optional[int]) – Number of boosting rounds.
            max_leaves – Maximum number of leaves; 0 indicates no limit.
            min_child_weight (Optional[float]) – Minimum sum of instance weight(hessian) needed in a child.
            learning_rate (Optional[float]) – Boosting learning rate (xgb's “eta”)
            subsample (Optional[float]) – Subsample ratio of the training instance.
            colsample_bytree (Optional[float]) – Subsample ratio of columns when constructing each tree.
            colsample_bylevel (Optional[float]) – Subsample ratio of columns for each level.
            reg_alpha (Optional[float]) – L1 regularization term on weights (xgb's alpha).
            reg_lambda (Optional[float]) – L2 regularization term on weights (xgb's lambda).

            We have a budget to try 10 configurations in total. You will get the validation error rate (1- accuracy) before you need to specify the next configuration.
            The goal is to find the configuration that minimizes the error rate with the given budget, so you should explore different parts of the search space if the loss is not changing.
            Provide a config in JSON format, enclosed in triple backticks. Do not put new lines or any extra characters in the response. Example config: {"C": x, "gamma": y}
            """
        if self.classifier_type == "svc":
            content = """You are helping tune hyperparameters for a SVM model. Training is done with the sklearn library in python. This is our hyperparameter search space:
            C: float, - Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty. Range: [0.001, 1000]
            kernel {'poly', 'rbf', 'sigmoid'}  - Specifies the kernel type to be used in the algorithm. 

            We have a budget to try 10 configurations in total. You will get the validation error rate (1- accuracy) before you need to specify the next configuration.
            The goal is to find the configuration that minimizes the error rate with the given budget, so you should explore different parts of the search space if the loss is not changing.
            Provide a config in JSON format, enclosed in triple backticks. Do not put new lines or any extra characters in the response. Example config: {"C": x, "gamma": y}
            """
        system_content = "You are a machine learning expert"
        system_prompt = {"role": "system", "content": system_content}
        prompt = {"role": "user", "content": content}
        assistant_response = {"role": "assistant", "content": "Config: "}

        messages = [system_prompt, prompt, assistant_response]
        result = self.llm.create(
                    engine=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=500,
                )
        result = result['choices'][0]['message']['content']
        messages = messages[:-1] + [{"role": "assistant", "content": result}]
        try:
            config = json.loads(result)
        except json.decoder.JSONDecodeError:
            result = result.replace("```json", "```")
            action_match = re.search(R"```(.*?)```?", result, re.DOTALL)
            match_str = action_match.group(1).strip() if action_match else result
            config = cast(dict, json.loads(match_str, strict=False)) if match_str else {}
        return config, messages

    def optimize_hyperparamters(self, loss: float, messages: list) -> tuple[dict, list]:
        """
        Optimize hyperparameters.

        Args:
            loss (any): The loss value.
            messages (list): The list of messages.

        Returns:
            tuple[dict, list]: A tuple containing a dictionary with the configuration and a list of messages.
        """
        current_messages = copy.deepcopy(messages)
        current_content = f"""loss = { loss :.4e}. Specify the next config."""
        current_prompt = {"role": "user", "content": current_content}
        assistant_response = {"role": "assistant", "content": "Config: "}
        current_messages = current_messages + [current_prompt, assistant_response]
        #current_messages = current_messages + [current_prompt]
        result = self.llm.create(
                engine=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=500,
            )
        result = result['choices'][0]['message']['content']
        current_messages = current_messages[:-1] + [{"role": "assistant", "content": result}]
        #current_messages = current_messages + [{"role": "assistant", "content": result}]
        try:
            config = json.loads(result)
        except json.decoder.JSONDecodeError:
            result = result.replace("```json", "```")
            action_match = re.search(R"```(.*?)```?", result, re.DOTALL)
            match_str = action_match.group(1).strip() if action_match else result
            config = cast(dict, json.loads(match_str, strict=False)) if match_str else {}
        return config, current_messages

    def run_optimization(self, fout: str, iterations: int = 10) -> None:
        """
        Run optimization for a given number of iterations.

        Args:
            iterations (int, optional): The number of iterations to run the optimization. Default is 10.

        Returns:
            None
        """
        print("Initial config")
        config, messages = self.get_initial_config()
        print(config)
        loss = train_classifier(
            classifier_type=self.classifier_type,
            train_data=self.train,
            test_data=self.test,
            config=config
        )
        print(f"Initial loss: {loss}")
        print("-------------")
        self.configs.append(config)
        self.losses.append(loss)
        log = dict()
        list_of_logs = []
        log['iteration'] = 0
        log['config'] = config
        log['loss'] = float(loss)
        list_of_logs.append(log)
        for i in range(iterations-1):
            log = dict()
            print(f"Iteration: {i+1}")
            config, messages = self.optimize_hyperparamters(loss, messages)
            print(f"config: {config}")
            if config in self.configs:
                print("Repeated config, using previous loss")
                loss = self.losses[self.configs.index(config)]
            else:
                loss = train_classifier(
                    classifier_type=self.classifier_type,
                    train_data=self.train,
                    test_data=self.test,
                    config=config
                )
            self.configs.append(config)
            self.losses.append(loss)
            print(f"loss: {loss}")
            print("-------------")
            log['iteration'] = i+1
            log['config'] = config
            log['loss'] = float(loss)
            list_of_logs.append(log)
            #losses_random_search.append(losses_epoch)
        # write log to json
        with open(fout, 'w') as f:
            json.dump(list_of_logs, f, indent=4)   
        #return self.losses
