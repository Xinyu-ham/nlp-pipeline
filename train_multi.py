import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from torch.distributed import init_process_group, destroy_process_group
from transformers import AutoTokenizer
import yaml

from nlp_model.datapipe import NewsDataPipe
from nlp_model.model import Trainer

import optuna

from datetime import datetime
import boto3, json


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
CONFIG_PATH = 'nlp_model/tuning.yml'
DISTRIBUTED = True

# read yml
with open(CONFIG_PATH, 'r') as f:
    experiments = yaml.safe_load(f) 
tuning_config = experiments['experiments'][0]

def train_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
        print('Using GPU..')
    else:
        print('Using CPU..')
    trainer = Trainer(tuning_config, device, tuning_config['id'] + datetime.now().strftime("%Y%m%d%H%M%S"), DISTRIBUTED)


    print(f'Training on {TRAIN_FILES} batches..')
    def objective(trial: optuna.Trial):
        return trainer.train(trial, TRAIN_S3_URL, TRAIN_DATASET_SIZE, TEST_S3_URL, TEST_DATASET_SIZE)
    
    study = optuna.create_study(direction=tuning_config['objective'], study_name=tuning_config['name'])
    study.optimize(objective, n_trials=tuning_config['n_trials'])

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

if __name__ == '__main__':
    init_process_group(backend='gloo')
    train_model()
    destroy_process_group()
    # print(tuning_config)



