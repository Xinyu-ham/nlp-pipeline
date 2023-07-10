import boto3, json, os

class Environment:
    BUCKET_NAME = 'xy-mp-pipeline'
    AWS_REGION = 'ap-southeast-1'
    bucket = boto3.resource('s3', region_name=AWS_REGION).Bucket(BUCKET_NAME)
    MODEL_METADATA = json.loads(bucket.Object('data/covid-csv-metadata.json').get()['Body'].read())
    OUTPUT_PATH = 'data/covid-csv'
    N_SAMPLES = MODEL_METADATA['dataset_size']
    BATCH_SIZE = MODEL_METADATA['batch_size']
    TRAIN_FILES = N_SAMPLES * 4 // 5 // BATCH_SIZE + 1
    TEST_FILES = N_SAMPLES // 5 // BATCH_SIZE + 1
    TRAIN_S3_URL = f's3://{BUCKET_NAME}/{OUTPUT_PATH}/training/'
    TEST_S3_URL = f's3://{BUCKET_NAME}/{OUTPUT_PATH}/testing/'
    TEST_DATASET_SIZE = MODEL_METADATA['test_size']
    TRAIN_DATASET_SIZE = MODEL_METADATA['train_size']
    MODEL_OUTPUT_PATH = 'assets/model'
    CONFIG_PATH = 'nlp_model/tuning.yaml'
    MODEL_S3_PATH = 's3://xy-mp-pipeline/artifects/model/fake_news_model.pt'
    DISTRIBUTED = True
    MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'http://0.0.0.0:5000')