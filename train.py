import copy
import logging
import os

import torch


def validate(model, loss_fn, eval_func, val_loader, config):
    model.eval()
    with torch.no_grad():
        val_loss, val_eval = {}, {}
        for r in config.eval.results:
            val_eval[r] = []
        for r in config.loss.results:
            val_loss[r] = []

        for batch in val_loader:
            batch = batch.to(device=config.run.device)
            output = model(batch)
            loss = loss_fn(output, batch)
            eval = eval_func(output, batch)

            for r in config.loss.results:
                val_loss[r].append(loss[r].item())
            for r in config.eval.results:
                val_eval[r].append(eval[r])

    for r in config.eval.results:
        val_eval[r] = sum(val_eval[r]) / len(val_eval[r])
    for r in config.loss.results:
        val_loss[r] = sum(val_loss[r]) / len(val_loss[r])

    return val_loss, val_eval


def train(model, loss_fn, eval_fn, train_loader, val_loader, config, root_path):
    best_model = copy.deepcopy(model.state_dict())
    best_eval = float('-inf')

    optimizer = torch.optim.Adam(model.parameters(), lr=config.run.learning_rate, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.run.learning_rate_decay_step,
                                                gamma=config.run.learning_rate_decay)

    logging.info("Training begin ...")

    # TODO: here the batch and iteration problem
    try:
        for i, batch in zip(range(1, len(train_loader) + 1), train_loader):
            model.train()
            batch = batch.to(device=config.run.device)
            output = model(batch)
            loss = loss_fn(output, batch)

            optimizer.zero_grad()
            loss['loss'].backward()
            optimizer.step()
            scheduler.step(i)

            if i % config.run.print_per_iter != 0:
                continue

            val_loss, val_eval = validate(model, loss_fn, eval_fn, val_loader, config)

            best = '*' if val_eval['eval'] > best_eval else ''
            if val_eval['eval'] > best_eval:
                best_eval, best_model = val_eval['eval'], copy.deepcopy(model.state_dict())

            for r in config.loss.results:
                config.tensorboard.add_scalar(f'valid/loss/{r}', val_loss[r], i)
            for r in config.eval.results:
                config.tensorboard.add_scalar(f'valid/eval/{r}', val_eval[r], i)

            val_loss_msg = ' '.join('{}: {:.4f}'.format(k, v) for k, v in val_loss.items())
            val_eval_msg = ' '.join('{}: {:.4f}'.format(k, v) for k, v in val_eval.items())

            msg = "Iter: [{}/%d] val [{}] [{}] best [{:.4f}] {}" % config.run.total_iter
            logging.info(msg.format(i, val_loss_msg, val_eval_msg, best_eval, best))

            if val_eval['eval'] >= best_eval:
                torch.save(model.state_dict(), os.path.join(root_path, 'model.pt'))

    except KeyboardInterrupt:
        logging.info('KeyboardInterrupt best val acc: {:.4f}'.format(best_eval))
    except Exception as e:
        logging.exception(e)

    logging.info('Train end best val acc: {:.4f}'.format(best_eval))

