import torch
import optuna
from optuna.integration.mlflow import MLflowCallback
from torch.distributed import init_process_group, destroy_process_group

from nlp_model.model import Trainer
from constants import Environment as E

import boto3, json, yaml
from datetime import datetime

import mlflow
import mlflow.pytorch

# read config
with open(E.CONFIG_PATH, 'r') as f:
    experiments = yaml.safe_load(f) 
tuning_config = experiments['experiments'][0]


def train_model() -> None:
    '''
    Train the model with the given config across multiple nodes and GPUs.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
        print('Using GPU..')
    else:
        print('Using CPU..')
    
    experiment_id = tuning_config['id'] + datetime.now().strftime("%Y%m%d%H%M%S")
    trainer = Trainer(tuning_config, device, experiment_id, E.DISTRIBUTED)

    print(f'Training on {E.TRAIN_FILES} batches..')

    def objective(trial: optuna.Trial) -> float:
        '''
        Objective function for optuna to optimize.

        Args:
            trial: optuna.Trial object

        Returns:
            float: the objective value
        '''
        mlflow.set_tracking_uri(E.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_id)
        with mlflow.start_run(run_name=f'trial_{trial.number}'):
            acc = trainer.train(trial, E.TRAIN_S3_URL, E.TRAIN_DATASET_SIZE, E.TEST_S3_URL, E.TEST_DATASET_SIZE)
            mlflow.log_params(trial.params)
            mlflow.log_metrics({'validation acc': acc})
            mlflow.pytorch.log_model(trainer.model, f'model')
        return acc
    
    study = optuna.create_study(direction=tuning_config['objective'], study_name=tuning_config['name'], pruner=optuna.pruners.HyperbandPruner(max_resource='auto'))
    study.optimize(
        objective, 
        n_trials=tuning_config['n_trials'], 
    )

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)


if __name__ == '__main__':
    init_process_group(backend='gloo')
    train_model()
    destroy_process_group()




