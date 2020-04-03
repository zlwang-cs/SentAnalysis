import logging
import os
from functools import reduce

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from utils.tokenizer import ParaSplit
from transformers import *


class Batch(object):

    def __init__(self, **kwargs):
        """ initialize all input """
        for key, value in kwargs.items():
            self[key] = value

    def __setitem__(self, key, value):
        """ sets the attribute """
        setattr(self, key, value)

    def __getitem__(self, key):
        """ gets the data of the attribute """
        return getattr(self, key, None)

    def to(self, device):
        """ change tensor device """
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                self[key] = self[key].to(device)
        return self


class MyDataset(Dataset):
    def __init__(self, name, config):
        super(MyDataset, self).__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.model.bert.weight_dir)

        self.data = self.load(name)

    def load(self, name):
        ret = []
        # sent_tokenizer = ParaSplit()
        max_sentence_length = self.config.model.bert.config.max_position_embeddings
        with open(os.path.join(self.config.dataset.path, name+'.txt')) as fin:
            for line in fin:
                text, label = line.strip().rsplit('\t', 1)
                # sent = sent_tokenizer.split(text)
                # sent = [self.tokenizer.encode(s, max_length=max_sentence_length) for s in sent]
                sent = self.tokenizer.encode(text, max_length=max_sentence_length)
                label = int(label)
                ret.append((sent, label))
        return ret

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    sentences, label = zip(*batch)
    sentences, label = list(sentences), list(label)

    # sent_num = [len(s) for s in sentences]
    # sentences = reduce(lambda x, y: x + y, sentences)
    sentences = list(map(lambda x: torch.tensor(x, dtype=torch.long), sentences))
    sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True)

    attention_masks = (sentences != 0).long()

    label = torch.tensor(label, dtype=torch.long)

    return Batch(sentence=sentences, mask=attention_masks, label=label)
    # return Batch(sentence=sentences, mask=attention_masks, sent_num=sent_num, label=label)


class PageSampler(Sampler):

    def __init__(self, data_source, batch_size, iteration, shuffle=True):
        super(PageSampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_size = batch_size
        self.iteration = iteration
        self.shuffle = shuffle

    def idx_iter(self):
        n = len(self.data_source)
        if self.shuffle:
            idxs = torch.randperm(n).tolist()
        else:
            idxs = list(range(n))
        for idx in idxs:
            yield idx

    def __iter__(self):
        # yield [721, 316, 240]
        batch = []
        idx_iter = self.idx_iter()
        for i in range(self.iteration):
            while len(batch) < self.batch_size:
                try:
                    new_idx = next(idx_iter)
                except StopIteration:
                    idx_iter = self.idx_iter()
                    new_idx = next(idx_iter)
                if new_idx not in batch:
                    batch.append(new_idx)
            assert len(batch) == self.batch_size
            # print(batch)
            yield batch
            batch = []

    def __len__(self):
        return self.iteration


def load_data(config):
    logging.info('Loading data begin')

    train_dataset = MyDataset('train', config)
    test_dataset = MyDataset('test', config)

    train_loader = DataLoader(train_dataset,
                              pin_memory=True, collate_fn=collate_fn,
                              batch_sampler=PageSampler(train_dataset,
                                                        config.run.batch_size,
                                                        iteration=config.run.total_iter,
                                                        shuffle=True)
                              )

    test_loader = DataLoader(test_dataset, batch_size=config.run.batch_size, pin_memory=True, collate_fn=collate_fn)

    return train_loader, test_loader
