
from .model import FakeNewsModel
import spacy
from collections import Counter
import re
import torch
from transformers import AutoTokenizer

class FakeNewsInference(FakeNewsModel):
    def __init__(self, model_file: str, metadata: dict):
        model_data = torch.load(model_file)
        pretrain_model = model_data['pretrain_model_name']
        model_states = model_data['model_states']
        super().__init__(pretrain_model, 0, 0, 0)
        self.model.load_state_dict(model_states)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
        self.nlp = spacy.load('en_core_web_sm')
        self.input_schema = metadata['schema']['input_features']

    def predict(self, headline: str) -> float:
        bert_input, tabular_input = self._preprocess_headline(headline)
        with torch.no_grad():
            output = self.model(bert_input, tabular_input)
        return output.item()

    def _preprocess_headline(self, headline: str) -> tuple[dict, torch.Tensor]:
        tokens = self.tokenizer(headline, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
        tokens = {k: v.squeeze() for k, v in tokens.items()}
        input_ids = tokens['input_ids'].unsqueeze(0)
        attention_mask = tokens['attention_mask'].unsqueeze(0)
        bert_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'return_dict': False
        }

        headline_len = len(headline.split())
        has_stats = int(re.match(r'\d', headline) is not None)

        ner_count = [f'ner_{ent.label_}' for ent in self.nlp(headline).ents]
        ner_count = Counter(ner_count)
        ner_input = self._get_ner_input(ner_count, self.input_schema)
        return bert_input, torch.cat([torch.tensor([headline_len, has_stats]), ner_input], dim=0).unsqueeze(0)

    def _get_ner_input(ner_count: Counter, input_schema: dict) -> torch.Tensor:
        ner_features = [s for s in input_schema if s.startswith('ner_')]
        ner_input = torch.zeros(len(ner_features))
        for i, feat in enumerate(ner_features):
            ner_input[i] = ner_count.get(feat, 0)
        return ner_input