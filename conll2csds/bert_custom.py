import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, AlbertPreTrainedModel, AlbertModel, DistilBertPreTrainedModel, DistilBertModel

class BertEFP(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, 256, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(256 * 2, 1)

    def forward(self, input_ids, attention_mask):
        sequence_output, pooled_output = self.bert(input_ids, attention_mask=attention_mask, return_dict = False)
        lstm_output, (h, c) = self.lstm(sequence_output)
        hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)
        linear_output = self.linear(hidden.view(-1,
                                                256 * 2))

        return linear_output
