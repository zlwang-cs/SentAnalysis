from functools import reduce

import torch
import torch.nn as nn

from transformers import *


class BertBasedModel(nn.Module):
    def __init__(self, config):
        super(BertBasedModel, self).__init__()
        self.config = config

        self.bert_layer = BertModel.from_pretrained(config.model.bert.weight_dir)
        if config.model.bert.freeze:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(input_size=config.model.bert.config.hidden_size, hidden_size=config.model.lstm.hidden_size,
                            bidirectional=True)

        self.fc = nn.Sequential(nn.Linear(config.model.lstm.hidden_size * 2, config.model.fc),
                                nn.ReLU(),
                                nn.Linear(config.model.fc, config.dataset.category))

    def forward(self, batch):
        sent = batch.sentence
        mask = batch.mask
        sent_num = batch.sent_num

        bert_last_hidden = self.bert_layer(sent, attention_mask=mask)[0]
        bert_cls_hidden = bert_last_hidden[:, 0, :]

        output = []
        start = 0
        for n in sent_num:
            feat = bert_cls_hidden[start, start + n]
            # max_feat, _ = feat.max(0)
            feat = self.lstm(feat)
            output.append(self.fc(feat))

            start += n

        output = torch.stack(output)

        return output
