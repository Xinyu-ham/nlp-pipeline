import pandas as pd
import torch, os
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from transformers import AutoTokenizer

class NewsDataPipe(IterDataPipe):
    '''
    DataPipe for loading news data from S3. Has an option to shard the data to different GPUs during distributed training.

    Args:
        s3_url (str): S3 URL to load data from
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use
        num_files (int): Number of files to load
        distributed (bool): Whether to shard the data to different GPUs during distributed training
    '''
    def __init__(self, s3_url: str, tokenizer: AutoTokenizer, num_files: int, distributed: bool=False) -> None:
        '''
        Attributes:
            url_wrapper (IterableWrapper): IterableWrapper object that wraps the S3 URL
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use
            num_files (int): Number of files to load
            distributed (bool): Whether to shard the data to different GPUs during distributed training
        '''
        super().__init__()
        self.url_wrapper = IterableWrapper([s3_url]).list_files_by_s3().shuffle().sharding_filter()
        self.tokenizer = tokenizer
        self.num_files = num_files
        self.distributed = distributed

    def __iter__(self):
        '''
        Yields:
            bert_input (list): List of BERT input tensors containing input_ids and attention_mask
            tabular_input (list): List of tabular input tensors
            label (torch.Tensor): Label tensor
        '''
        for i, (_, file) in enumerate(self.url_wrapper.load_files_by_s3()):
            temp = pd.read_csv(file)
            label = torch.from_numpy(temp['outcome'].values)
            # For BERT model
            bert_input = []
            embedded = [self.tokenizer(t, padding='max_length', max_length=100, truncation=True, return_tensors='pt') for t in temp['headlines']]
            bert_input.append(torch.cat([e['input_ids'] for e in embedded], dim=0))
            bert_input.append(torch.cat([e['attention_mask'] for e in embedded], dim=0))

            # Tabular features
            tabular_input = [torch.from_numpy(temp[col].values).to(torch.float32).squeeze() for col in temp.columns if col not in ['outcome', 'headlines']]
            if self.distributed:
                # Sharding each file to different GPU
                global_rank = int(os.environ['RANK'])
                world_size = int(os.environ['WORLD_SIZE'])
                if i % world_size == global_rank:
                    if i % 100 == 0:
                        print(f'[GPU {global_rank}] |   Beginning batch {i}..')
                    yield bert_input, tabular_input, label
            else:
                # Sharding every file to one GPU
                yield bert_input, tabular_input, label

    def __len__(self):
        '''
        Returns:
            num_files (int): Number of files to load
        '''
        return self.num_files
    
