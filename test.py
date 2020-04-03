import argparse
import json
import logging
import os

import torch
import yaml
from easydict import EasyDict

import models.evaluation as eval
import models.loss as loss
from models.model import BertBasedModel
from utils.dataset import load_data


def test(model, loss_fn, eval_fn, val_loader, config):
    model.eval()
    with torch.no_grad():
        val_loss, val_eval = {}, {}
        for r in eval_fn.result:
            val_eval[r] = []
        for r in loss_fn.loss:
            val_loss[r] = []

        for i, batch in zip(range(1, len(val_loader) + 1), val_loader):
            batch = batch.to(device=config.run.device)
            output = model(batch)
            loss = loss_fn(output, batch)
            eval = eval_fn(output, batch)

            for r in loss_fn.loss:
                val_loss[r].append(loss[r].item())
            for r in eval_fn.result:
                val_eval[r].append(eval[r])

            msg = "Iter: [{}/%d] Test [{}] [{}]" % len(val_loader)
            loss_msg = ' '.join('{}: {:.4f}'.format(k, v) for k, v in loss.items())
            eval_msg = ' '.join('{}: {:.4f}'.format(k, v) for k, v in eval.items())
            logging.info(msg.format(i, loss_msg, eval_msg))

    for r in eval_fn.result:
        val_eval[r] = sum(val_eval[r]) / len(val_eval[r])
    for r in loss_fn.loss:
        val_loss[r] = sum(val_loss[r]) / len(val_loss[r])

    return val_loss, val_eval


def get_bert_config(bert_path):
    bert_config_path = os.path.join(bert_path, "config.json")
    bert_config = json.load(open(bert_config_path))

    return EasyDict(bert_config)


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

    logger_path = f'TEST-[{args.dir}]-[{args.weight}]'

    log.setLevel(logging.INFO)
    os.makedirs(os.path.join('log', logger_path))

    file_path = os.path.join('log', logger_path, 'info.log')
    file_handler = logging.FileHandler(file_path, mode='w')
    file_handler.setFormatter(log_format)
    log.addHandler(file_handler)


model_dict = {'BertBasedModel': BertBasedModel}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', dest='dir', type=str)
    parser.add_argument('--weight', dest='weight', default='model.pt', type=str)
    args = parser.parse_args()

    check_dir = os.path.join('log', args.dir)
    checkpoint = os.path.join(check_dir, args.weight)
    with open(os.path.join(check_dir, 'config.yaml'), 'r', encoding='utf8') as fc:
        config = EasyDict(yaml.load(fc))
    set_logger()

    config.model.bert.config = get_bert_config(config.model.bert.weight_dir)

    logging.info('Running begin!')

    model = model_dict[config.model.name](config).to(torch.device(config.run.device))
    model.load_state_dict(torch.load(checkpoint))

    loss_fn = getattr(loss, config.loss.func)(config)
    eval_fn = getattr(eval, config.eval.func)(config)

    train_loader, val_loader = load_data(config)
    val_loss, val_eval = test(model, loss_fn, eval_fn, val_loader, config)

    print(val_loss)
    print(val_eval)
