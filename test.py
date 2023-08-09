#!/bin/python3
'''
    This is a sample training script to test if script works in K8 cluster
'''
import torch, os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='whether to use GPU')
    return parser.parse_args()

class MyDataPipe(IterDataPipe):
    def __init__(self, s3_url):
        super().__init__()
        self.url_wrapper = IterableWrapper([s3_url]).list_files_by_s3().shuffle().sharding_filter() 

    def __iter__(self):
        for i, (_, file) in enumerate(self.url_wrapper.load_files_by_s3()):
            temp = pd.read_csv(file).astype(np.float32)
            X = torch.from_numpy(temp.iloc[:, :-1].values)
            y = torch.from_numpy(temp['y'].values)
            
            global_rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            if i % world_size == global_rank:
                if i % 100 == 0:
                    print(f'[Device {global_rank}] |   Beginning batch {i}..')
                yield X, y

    def __len__(self):
        return 100

class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        l1 = self.linear1(x)
        output = self.sigmoid(l1)
        return output.squeeze(2)
    
def train(gpu):
    device_id = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{device_id}' if gpu else f'cpu:{device_id}'

    model = TestModel().to(device)
    if gpu:
        model = DDP(model, device_ids=[device_id], output_device=device_id)
    else:
        model = DDP(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    datapipe = MyDataPipe('s3://xy-mp-pipeline/data/test_dataset/')
    dataloader = DataLoader(datapipe, batch_size=5, shuffle=True)

    for epoch in range(3):
        print(f'[{device.upper()} {device_id}] | Starting Epoch {epoch}..')
        for X, y in dataloader:
            X = X.squeeze(2).to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(X) 
            
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    init_process_group(backend='gloo')
    gpu = parse_args().gpu
    train(gpu)
    destroy_process_group()

    

