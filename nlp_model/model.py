import optuna, os
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from .datapipe import NewsDataPipe

from transformers import logging
logging.set_verbosity_error()

class Trainer:
    '''
    Trainer class for training the model given a set of hyperparameters. Given an option to train in distributed mode.

    Args:
        config (dict): Dictionary containing the hyperparameters to be tuned.
        device (str): Device to train the model on.
        study_name (str): Name of the optuna study.
        distributed (bool, optional): Whether to train in distributed mode. Defaults to False.
    '''
    def __init__(self, config: dict, device: str, study_name: str, distributed: bool=False):
        '''
        Attributes:
            device (str): Device to train the model on.
            model (FakeNewsModel): Model to be trained.
            optimizer (torch.optim): Optimizer to be used for training.
            experiment_id (str): ID of the optuna study.
            experiment_name (str): Name of the optuna study.
            parameters (dict): Dictionary containing the hyperparameters to be tuned.
            distributed (bool): Whether to train in distributed mode.
            gpu_id (int): ID of the GPU to train on.
            suggestions (dict): Dictionary containing the hyperparameters to be tuned and their suggested values.
            epoch (int): Current epoch of training.
        '''
        self.device = device
        self.model = None 
        self.optimizer = None
        self.experiment_id = study_name
        self.experiment_name = config['name']
        self.parameters = config['parameters']
        self.distributed = distributed
        if distributed:
            self.gpu_id = os.environ['LOCAL_RANK']
            self.device = 'cuda:' + self.gpu_id
        else:
            self.gpu_id = 0
        print(f'[GPU {self.gpu_id}] | Allocated memory: {torch.cuda.memory_allocated(self.device)} bytes.')
        self.suggestions = {}
        self.epoch = 0
        self.location = './assets/models/' + self.experiment_id

    def build_model(self, trial: optuna.Trial) -> torch.nn.Module:
        '''
        Builds the model given a set of hyperparameters.
        
        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            torch.nn.Module: Model to be trained.
        '''
        for parameter in self.parameters:
            self.suggestions[parameter['name']] = self.map_parameters_to_suggestion(parameter, trial)
        model_params = ['dropout1', 'dropout2', 'hidden_size', 'pretrained_model']
        self.model = FakeNewsModel(**{model_param: self.suggestions[model_param] for model_param in model_params})
        return self.model

    def save_checkpoint(self, model_path: str) -> None:
        '''
        Saves the model to a given location. Saves the current epoch and optimizer state as well.

        Args:
            location (str): Location to save the model to.
        '''
        checkpoint = {
            'epoch': self.epoch, 
            'optimizer_states': self.optimizer.state_dict(),
            'pretrained_model': self.suggestions['pretrained_model'],
        }
        if self.distributed:
            model_states = self.model.module.state_dict()
            checkpoint['model_states'] = {'module.' + k:v for k, v in model_states.items()}
        else:
            checkpoint['model_states'] = self.model.state_dict()

        save_dir = os.path.dirname(model_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(checkpoint, model_path)
        print(f'[GPU {self.gpu_id}] | Completed epoch: {self.epoch} - Model saved to {self.location}')

    def load_checkpoint(self, model_path):
        '''
        Loads the model from a given location. Loads the current epoch and optimizer state as well.

        Args:
            model_path (str): Location to load the model from.
        '''
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model_states'])
            self.model.eval()
            self.optimizer.load_state_dict(checkpoint['optimizer_states'])
            print(f'[GPU {self.gpu_id}] | Loaded model from {model_path}.')


    def train(self, trial: optuna.Trial, train_url: str, train_len: int, test_url:str, test_len: int) -> float:
        '''
        Trains the model given a set of hyperparameters.

        Args:
            trial (optuna.Trial): Optuna trial object.
            train_url (str): AWS S3 URL of the training dataset.
            train_len (int): Length of the training dataset.
            test_url (str): AWS S3 URL of the test dataset.
            test_len (int): Length of the test dataset.

        Returns:
            float: Accuracy of the model on the test dataset.
        '''
        self.model = self.build_model(trial).to(self.device)
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.device])
        tokenizer = AutoTokenizer.from_pretrained(self.suggestions['pretrained_model'])

        train_data = NewsDataPipe(train_url, tokenizer, train_len, self.distributed)
        test_data = NewsDataPipe(test_url, tokenizer, test_len, self.distributed)

        train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=1, shuffle=True)


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.suggestions['learning_rate'])
        loss_function = torch.nn.BCELoss().to(self.device)

        for _ in range(self.epoch, self.suggestions['epochs']):
            # torch.cuda.empty_cache()
            # self.load_checkpoint(f'{self.location}/trial_{trial.number}.pt')
            self._run_epoch(train_loader, loss_function)
            self.epoch += 1
            # if int(self.gpu_id) == 0:
            #     self.save_checkpoint(f'{self.location}/trial_{trial.number}.pt')
        return self._evaluate_validation_set(test_loader)

    def _evaluate_validation_set(self, test_loader: DataLoader) -> float:
        '''
        Evaluates the model accuracy on the validation set.

        Args:
            test_loader (DataLoader): PyTorch DataLoader object for the validation set.

        Returns:
            float: Accuracy of the model on the validation set.
        '''
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

    def _run_epoch(self, train_loader: DataLoader, loss_function: torch.nn.Module) -> None:
        '''
        Runs a single epoch of training.

        Args:
            train_loader (DataLoader): PyTorch DataLoader object for the training set.
            loss_function (torch.nn.Module): PyTorch loss function.
        '''
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
            self.optimizer.step()


    @staticmethod
    def map_parameters_to_suggestion(parameters: dict, trial: optuna.Trial):
        '''
        Maps the parameters to the optuna suggestion.

        Args:
            parameters (dict): Dictionary of parameters.
            trial (optuna.Trial): Optuna trial object.

        Returns:
            Any: Optuna suggestion.
        '''
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
    '''
    A multi-modal network for Covid fake news detection, using a pretrained transformer model and tabular data.

    Args:
        pretrained_model (str): Pretrained model name.
        dropout1 (float): Dropout rate for the first dropout layer.
        dropout2 (float): Dropout rate for the second dropout layer.
        hidden_size (int): Size of the hidden layer.
    '''
    def __init__(self, pretrained_model: str, dropout1: float=0.25, dropout2: float=0.25, hidden_size: int=12) -> None:
        '''
        Initializes the model.
        
        Attributes:
            pretrained_model (str): Pretrained transformer model name.
            dropout1 (float): Dropout rate for the first dropout layer.
            linear1 (torch.nn.Linear): Linear layer after the first dropout layer.
            dropout2 (float): Dropout rate for the second dropout layer.
            linear2 (torch.nn.Linear): Linear layer after the second dropout layer.
            normalize (torch.nn.functional.normalize): Normalization function.
            relu (torch.nn.ReLU): ReLU activation function.
            sigmoid (torch.nn.Sigmoid): Sigmoid activation function.
        '''
        # define layers
        super(FakeNewsModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout_1 = torch.nn.Dropout(dropout1)
        self.linear_1 = torch.nn.Linear(384, hidden_size)
        self.dropout_2 = torch.nn.Dropout(dropout2)
        self.linear_2 = torch.nn.Linear(hidden_size + 20, 1)
        self.normalize = torch.nn.functional.normalize
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, bert_input: dict, tabular_input: list) -> torch.Tensor:
        '''
        Forward pass of the model.

        Args:
            bert_input (dict): Dictionary of input_ids, attention_mask, and return_dict.
            tabular_input (list): List of tabular data.

        Returns:
            torch.Tensor: Tensor of logits from sigmoid.
        '''
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
    