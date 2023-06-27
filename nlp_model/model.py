import optuna
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, AutoTokenizer
from .datapipe import NewsDataPipe

from transformers import logging
logging.set_verbosity_error()

class Trainer:
    '''
    '''
    def __init__(self, config: dict, device):
        self.device = device
        self.model = None 
        self.experiment_id = config['id']
        self.experiment_name = config['name']
        self.parameters = config['parameters']
        self.suggestions = {}
        self.epoch = 0

    def build_model(self, trial: optuna.Trial):
        for parameter in self.parameters:
            self.suggestions[parameter['name']] = self.map_parameters_to_suggestion(parameter, trial)
        model_params = ['dropout1', 'dropout2', 'hidden_size', 'pretrained_model']
        self.model = FakeNewsModel(**{model_param: self.suggestions[model_param] for model_param in model_params})
        return self.model

    def save_model(self):
        pass

    def load_model(self):
        pass

    def train(self, trial: optuna.Trial, train_url: str, train_length: int, test_url:str, test_len: int):
        self.model = self.build_model(trial).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.suggestions['pretrained_model'])

        train_data = NewsDataPipe(train_url, tokenizer, train_length)
        test_data = NewsDataPipe(test_url, tokenizer, test_len)

        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True)


        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.suggestions['learning_rate'])
        loss_function = torch.nn.BCELoss()

        self.epoch = 0
        for _ in range(self.suggestions['epochs']):
            self._run_epoch(train_loader, loss_function, optimizer)

        
        return self._evaluate_validation_set(test_loader)

    def _evaluate_validation_set(self, test_loader: DataLoader):
        validation_acc = 0
        with torch.no_grad():
            for bert_input, tabular_input, label in test_loader:
                bert_input = {
                    'input_ids': bert_input[0].squeeze().to(self.device),
                    'attention_mask': bert_input[1].squeeze().to(self.device),
                    'return_dict': False
                }
                tabular_input = torch.cat(tabular_input).T.to(self.device)
                label = label.T.to(self.device)

                output = self.model(bert_input, tabular_input)
                # get acc of signmoid output
                acc = (output[0].round() == label).sum().item()
                validation_acc += acc
        return validation_acc / len(test_loader)

    def _run_epoch(self, train_loader: DataLoader, loss_function, optimizer):
        training_loss = 0
        training_acc = 0
        

        for bert_input, tabular_input, label in train_loader:
            bert_input = {
                'input_ids': bert_input[0].squeeze().to(self.device),
                'attention_mask': bert_input[1].squeeze().to(self.device),
                'return_dict': False
            }
            tabular_input = torch.cat(tabular_input).T.to(self.device)
            label = label.T.to(self.device)

            output = self.model(bert_input, tabular_input)

            loss = loss_function(output, label.float())
            training_loss += loss.item()

            # get acc of signmoid output
            acc = (output[0].round() == label).sum().item()
            training_acc += acc

            self.model.zero_grad()
            loss.backward()
            optimizer.step()


    @staticmethod
    def map_parameters_to_suggestion(parameters: dict, trial: optuna.Trial):
        if parameters['type'] == 'choice':
            return trial.suggest_categorical(parameters['name'], parameters['choices'])
        elif parameters['type'] == 'float' and parameters['scalingType'] == 'Descrete':
            return trial.suggest_float(parameters['name'], parameters['minValue'], parameters['maxValue'], step=parameters['q'])
        elif parameters['type'] == 'float' and parameters['scalingType'] == 'Logarithmic':
            return trial.suggest_float(parameters['name'], parameters['minValue'], parameters['maxValue'], log=True)
        elif parameters['type'] == 'float' and parameters['scalingType'] == 'Linear':
            return trial.suggest_float(parameters['name'], parameters['minValue'], parameters['maxValue'])
        elif parameters['type'] == 'int':
            return trial.suggest_int(parameters['name'], parameters['minValue'], parameters['maxValue'])
        else:
            raise ValueError(f'Invalid parameter type: {parameters["type"]}')
        


class FakeNewsModel(torch.nn.Module):
    def __init__(self, pretrained_model: str, dropout1: float=0.25, dropout2: float=0.25, hidden_size: int=12):
        # define layers
        super(FakeNewsModel, self).__init__()
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
    