# Using Large Language Models for Hyperparamter Optimization
This repo demonstrates how to use GPT-4 Turbo to optimize hyperparamter of tradionall models like SVMs or XGBoost on the basis of a text classification task. The details can be found in this artice.

## Getting started

Copy config/config_template.yaml into config/config.yaml and set the variables with your own values before running the samples.
Add your api key to .env `APIKEY=...`

### Using a virtual environment

Install Python 3.10 and create a virtual environment named "venv" using
`python -m venv venv`
and install the required modules by first starting the virtual environment using
`venv\scripts\activate`
and then
`pip install -r requirements.txt`.

## Running

The jupyter-notebook main.ipynb provides an exmaple on how to run the LLM HPO.



