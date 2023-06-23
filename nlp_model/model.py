import optuna
import torch
from transformers import BertModel

class ModelConstructor:
    '''
    '''
    def __init__(self, config: dict, train_data_url: str, test_data_url: str, device: str='cpu'):
        self.device = device
        self.model = None 
        self.train_data_url = train_data_url
        self.test_data_url = test_data_url
        self.experiment_id = config['experiment_id']
        self.experiment_name = config['experiment_name']
        self.parameters = config['parameters']
        self.suggestions = self._get_suggestions()

    def build_model(self, trial: optuna.Trial):
        model_params = ['dropout', 'hidden_size', 'pretrained_model']
        self.model = FakeNewsModel(**{param: self.suggestions[param] for param in model_params})
        return self.model

    def save_model(self):
        pass

    def load_model(self):
        pass

    def train(self, trial: optuna.Trial):
        model = self.build_model(trial)
        

    def evaluate(self):
        pass

    def upload_model(self):
        pass

    @staticmethod
    def map_parameters_to_suggestion(parameters: dict, trial: optuna.Trial):
        if parameters['type'] == 'choice':
            return trial.suggest_catrgorical(parameters['name'], parameters['choices'])
        elif parameters['type'] == 'float' and parameters['scalingType'] == 'Descrete':
            return trial.suggest_discrete_uniform(parameters['name'], parameters['minValue'], parameters['maxValue'], parameters['q'])
        elif parameters['type'] == 'float' and parameters['scalingType'] == 'Logarithmic':
            return trial.suggest_loguniform(parameters['name'], parameters['minValue'], parameters['maxValue'])
        elif parameters['type'] == 'float' and parameters['scalingType'] == 'Linear':
            return trial.suggest_uniform(parameters['name'], parameters['minValue'], parameters['maxValue'])
        elif parameters['type'] == 'int':
            return trial.suggest_int(parameters['name'], parameters['low'], parameters['high'])
        else:
            raise ValueError(f'Invalid parameter type: {parameters["type"]}')
        


class FakeNewsModel(torch.nn.module):
    def __init__(self, pretrained_model: str, model_id: int, dropout1: float=0.25, dropout2: float=0.25, hidden_size: int=12):
        self.model_id = model_id
        # define layers
        self.bert = BertModel.from_pretrained(pretrained_model)
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
    