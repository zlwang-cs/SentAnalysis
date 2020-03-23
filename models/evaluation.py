import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics


class Eval(nn.Module):
    result = ['eval']

    def forward(self, feature, batch):
        raise NotImplementedError


class F1(Eval):
    result = ['acc', 'recall', 'f1', 'eval']

    def __init__(self, config):
        super(F1, self).__init__()
        self.config = config
        self.category = config.dataset.category

    def forward(self, feature, batch):
        label = np.array(batch.label.cpu())
        predict = np.array(feature.max(-1)[1].detach().cpu())

        acc = metrics.accuracy_score(y_true=label, y_pred=predict)
        recall = metrics.recall_score(y_true=label, y_pred=predict)
        f1 = metrics.f1_score(y_true=label, y_pred=predict)

        return {'acc': acc,
                'recall': recall,
                'f1': f1,
                'eval': f1}
