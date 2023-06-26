import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from transformers import AutoTokenizer
import yaml

from nlp_model.datapipe import NewsDataPipe
from nlp_model.model import Trainer

import optuna

from tqdm import tqdm

BUCKET_NAME = 'xy-mp-pipeline'
OUTPUT_PATH = 'data/covid-csv'
N_SAMPLES = 19454
TRAIN_FILES = N_SAMPLES * 4 // 5 // 16 + 1
TEST_FILES = N_SAMPLES // 5 // 16 + 1
BATCH_SIZE = 16
TRAIN_S3_URL = f's3://{BUCKET_NAME}/{OUTPUT_PATH}/training/'
TEST_S3_URL = f's3://{BUCKET_NAME}/{OUTPUT_PATH}/testing/'
TEST_DATASET_SIZE = N_SAMPLES // 5
TRAIN_DATASET_SIZE = N_SAMPLES - TEST_DATASET_SIZE
MODEL_OUTPUT_PATH = 'assets/model'
PRETRAINED_MODEL_NAME = 'bert-base-uncased'
CONFIG_PATH = 'nlp_model/tuning.yml'

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
# read yml
with open(CONFIG_PATH, 'r') as f:
    experiments = yaml.safe_load(f) 
tuning_config = experiments['experiments'][0]

def train_model():
    dp = NewsDataPipe(TRAIN_S3_URL, tokenizer, TRAIN_DATASET_SIZE)
    dl = DataLoader(dp, batch_size=BATCH_SIZE, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(tuning_config, device)


    def objective(trial: optuna.Trial):
        return trainer.train(trial, TRAIN_S3_URL, TRAIN_DATASET_SIZE, TEST_S3_URL, TEST_DATASET_SIZE)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective)

    print('Number of finished trials: ', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

if __name__ == '__main__':
    train_model()
    # print(tuning_config)



