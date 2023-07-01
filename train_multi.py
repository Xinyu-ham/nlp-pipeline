import torch
import optuna
from torch.distributed import init_process_group, destroy_process_group

from nlp_model.model import Trainer
from contants import Environment as E

import boto3, json, yaml
from datetime import datetime


# read yml
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
    trainer = Trainer(tuning_config, device, experiment_id, E.MODEL_S3_PATH, E.DISTRIBUTED)

    print(f'Training on {E.TRAIN_FILES} batches..')

    def objective(trial: optuna.Trial) -> float:
        '''
        Objective function for optuna to optimize.

        Args:
            trial: optuna.Trial object

        Returns:
            float: the objective value
        '''
        return trainer.train(trial, E.TRAIN_S3_URL, E.TRAIN_DATASET_SIZE, E.TEST_S3_URL, E.TEST_DATASET_SIZE)
    
    study = optuna.create_study(direction=tuning_config['objective'], study_name=tuning_config['name'])
    study.optimize(objective, n_trials=tuning_config['n_trials'])

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)
    upload_model_to_s3(trial.number, trainer, E.MODEL_S3_PATH)

    

def upload_model_to_s3(trial_no, trainer, s3_path):
    '''
    Upload the model to S3.

    Args:
        trainer: Trainer object
        s3_path: S3 path to upload the model to
        save_local: local path to save the model to. Defaults to ''.
    '''
    model_path = f'{trainer.location}/trial_{trial_no}.pt'
    E.bucket.upload_file(model_path, s3_path)
    

if __name__ == '__main__':
    init_process_group(backend='gloo')
    train_model()
    destroy_process_group()




