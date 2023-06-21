import optuna
import torch
from transformers import BertModel

param_type_to_suggestion_map = {
    'int': optuna.trial.suggest_int,
    'float': optuna.trial.suggest_float,
    'choice': optuna.trial.suggest_categorical,
    'loguniform': optuna.trial.suggest_loguniform,
    'discrete_uniform': optuna.trial.suggest_discrete_uniform
}

class ModelConstructor:
    '''
    '''
    def __init__(self, config: dict, device: str='cpu'):
        self.device = device
        self.model = None 
        self.experiment_id = config['experiment_id']
        self.experiment_name = config['experiment_name']
        self.parameters = config['parameters']


    def create_model(self, trial: optuna.Trial):
        pass

    def _get_suggestions(self):
        fun = param_type_to_suggestion_map[self.parameters['type']]


    def save_model(self):
        pass

    def load_model(self):
        pass

    def train(self, trial: optuna.Trial):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

    def upload_model(self):
        pass
        


class FakeNewsModel(torch.nn.module):
    def __init__(self, pretrained: str, model_id: int, dropout1: float=0.25, dropout2: float=0.25, hidden_size: int=12):
        self.model_id = model_id
        # define layers
        self.bert = BertModel.from_pretrained(pretrained)
        self.dropout_1 = torch.nn.Dropout(dropout1)
        self.linear_1 = torch.nn.Linear(768, hidden_size)
        self.dropout_2 = torch.nn.Dropout(dropout2)
        self.linear_2 = torch.nn.Linear(hidden_size + 20, 1)
        self.normalize = torch.nn.functional.normalize
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, bert_input: dict, tabular_input: list):
        _, pooled_output = self.bert(**bert_input)
        dropout_1_output = self.dropout_1(pooled_output)
        linear_1_output = self.linear_1(dropout_1_output)
        relu_output = self.relu(linear_1_output)
        norm1 = self.normalize(relu_output, p=2, dim=1)
        norm2 = self.normalize(tabular_input, p=2, dim=1)
        combined_output = torch.cat([norm1, norm2], dim=1)
        dropout_2_output = self.dropout_2(combined_output)
        linear_2_output = self.linear_2(dropout_2_output)
        return self.sigmoid(linear_2_output)
    