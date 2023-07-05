import torch
import optuna

from nlp_model.model import Trainer

import boto3, json, yaml
from datetime import datetime


BUCKET_NAME = 'xy-mp-pipeline'

bucket = boto3.resource('s3').Bucket(BUCKET_NAME)
metadata = json.loads(bucket.Object('data/covid-csv-metadata.json').get()['Body'].read())

OUTPUT_PATH = 'data/covid-csv'
N_SAMPLES = metadata['dataset_size']
BATCH_SIZE = metadata['batch_size']
TRAIN_FILES = N_SAMPLES * 4 // 5 // BATCH_SIZE + 1
TEST_FILES = N_SAMPLES // 5 // BATCH_SIZE + 1
TRAIN_S3_URL = f's3://{BUCKET_NAME}/{OUTPUT_PATH}/training/'
TEST_S3_URL = f's3://{BUCKET_NAME}/{OUTPUT_PATH}/testing/'
TEST_DATASET_SIZE = metadata['test_size']
TRAIN_DATASET_SIZE = metadata['train_size']
MODEL_OUTPUT_PATH = 'assets/model'
CONFIG_PATH = 'nlp_model/tuning.yaml'

# read yml
with open(CONFIG_PATH, 'r') as f:
    experiments = yaml.safe_load(f) 
tuning_config = experiments['experiments'][0]

def train_model() -> None:
    '''
    Train the model with the given config on a single node and GPU.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
        print('Using GPU..')
    else:
        print('Using CPU..')
    
    trainer = Trainer(tuning_config, device, tuning_config['id'] + datetime.now().strftime("%Y%m%d%H%M%S"))

    print(f'Training on {TRAIN_FILES} batches..')

    def objective(trial: optuna.Trial) -> float:
        '''
        Objective function for optuna to optimize.

        Args:
            trial: optuna.Trial object

        Returns:
            float: the objective value
        '''
        return trainer.train(trial, TRAIN_S3_URL, TRAIN_DATASET_SIZE, TEST_S3_URL, TEST_DATASET_SIZE)
    
    study = optuna.create_study(direction=tuning_config['objective'], study_name=tuning_config['name'])
    study.optimize(objective, n_trials=tuning_config['n_trials'])

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print('  Value: ', trial.value)

if __name__ == '__main__':
    train_model()





