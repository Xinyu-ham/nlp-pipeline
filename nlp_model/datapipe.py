import pandas as pd
import torch
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from transformers import AutoTokenizer

class NewsDataPipe(IterDataPipe):
    '''
    DataPipe for loading news data from S3

    Args:
        s3_url (str): S3 URL to load data from
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use
        num_files (int): Number of files to load
    '''
    def __init__(self, s3_url: str, tokenizer: AutoTokenizer, num_files: int):
        super().__init__()
        self.url_wrapper = IterableWrapper(s3_url).list_files_by_s3().shuffle().sharding_filter()
        self.tokenizer = tokenizer
        self.num_files = num_files

    def __iter__(self):
        for _, file in self.url_wrapper.load_files_by_s3():
            temp = pd.read_csv(file)
            label = torch.from_numpy(temp['outcome'].values)
            # For BERT model
            bert_input = []
            embedded = [self.tokenizer(t, padding='max_length', max_length=100, truncation=True, return_tensors='pt') for t in temp['headlines']]
            bert_input.append(torch.cat([e['input_ids'] for e in embedded], dim=0))
            bert_input.append(torch.cat([e['attention_mask'] for e in embedded], dim=0))

            # Tabular features
            tabular_input = [torch.from_numpy(temp[col].values).to(torch.float32).squeeze() for col in temp.columns if col not in ['outcome', 'headlines']]
            yield bert_input, tabular_input, label

    def __len__(self):
        return self.num_files
    
