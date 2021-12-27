import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig
from factbank2csds import read_conll_data

class FactbankLoader(Dataset):
    def __init__(self, file_path, max_len, tokenizer):
        self.text, self.labels = read_conll_data(file_path)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        sentence = self.text[item]
        label = self.labels[item]
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        if len(tokens) < self.max_len:
            tokens = tokens + ['[PAD]' for _ in range(self.max_len - len(tokens))]
        else:
            tokens = tokens[:self.max_len - 1] + ['[SEP]']
        # Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        # Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()
        return input_ids, attention_mask, label
