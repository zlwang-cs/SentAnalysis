import argparse
import json
import logging
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import yaml
from easydict import EasyDict
from tensorboardX import SummaryWriter

import models.evaluation as eval
import models.loss as loss
from models.model import BertBasedModel
from train import train
from utils.dataset import load_data


def set_random_seed(seed_value=123456):
    """ set random seed for recurrence
    include torch torch.cuda numpy random """
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)


def set_logger():
    """ setting console log and file log (only train mode) of
    root logger with info level, copying config file to same
    directory for recurrence, directory path: log/m-d-h-M-feature
    (remove directory if it exists), then use logging.debug /
    logging.info / ... in all python package
    Args:
        mode: run mode (train or test)
    Return:
        logger_path: log directory (for saving model and other data)
    """
    log_format = '[%(asctime)s] [%(levelname)s] [%(filename)s] [Line %(lineno)d] %(message)s'
    log_format = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')
    log = logging.getLogger()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    log.addHandler(stream_handler)

    time = datetime.now().strftime("%b%d-%H%M%S")
    sp_tag = '' if config.run.tag is None else f'-[{config.run.tag}]'
    tag = f'{config.model.name}' + sp_tag
    logger_path = '{}-{}'.format(tag, time)

    log.setLevel(logging.INFO)
    os.makedirs(os.path.join('log', logger_path))
    shutil.copy(args.config, os.path.join('log', logger_path, 'config.yaml'))

    file_path = os.path.join('log', logger_path, 'info.log')
    file_handler = logging.FileHandler(file_path, mode='w')
    file_handler.setFormatter(log_format)
    log.addHandler(file_handler)
    return os.path.join('log', logger_path)


def get_writer(folder, flush_secs=120):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    writer = SummaryWriter(folder, flush_secs=flush_secs)
    return writer


def get_bert_config(bert_path):
    bert_config_path = os.path.join(bert_path, "config.json")
    bert_config = json.load(open(bert_config_path))

    return EasyDict(bert_config)


model_dict = {'BertBasedModel': BertBasedModel}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='config/default.yaml', type=str)
    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf8') as fc:
        config = EasyDict(yaml.load(fc))

    set_random_seed(config.run.random_seed)
    root_path = set_logger()
    writer = get_writer(folder=os.path.join(root_path, "tensorboard"))
    config.tensorboard = writer
    config.model.bert.config = get_bert_config(config.model.bert.weight_dir)

    logging.info('Running begin!')

    model = model_dict[config.model.name](config).to(torch.device(config.run.device))

    loss_fn = getattr(loss, config.loss.func)(config)
    eval_fn = getattr(eval, config.eval.func)(config)

    train_loader, val_loader = load_data(config)
    train(model, loss_fn, eval_fn, train_loader, val_loader, config, root_path)

    writer.close()
