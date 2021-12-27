import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AlbertPreTrainedModel, AlbertModel, DistilBertPreTrainedModel, DistilBertModel
import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BertEFP(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear1 = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        sequence_output, pooled_output = self.bert(input_ids, attention_mask=attention_mask, return_dict = False)

        lstm_output, (h, c) = self.lstm(sequence_output)
        hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)
        linear_output = self.linear(hidden.view(-1,
                                                256 * 2))
        linear1_output = self.linear1(sequence_output[:, 0, :].view(-1, 768))  ## extract the 1st token's embeddings

        linear2_output = self.linear2(linear1_output)

        return linear2_output

