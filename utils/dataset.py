import logging
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()




def load_data(config):
    logging.info('Loading data begin')

    all_pages = json.load(open(os.path.join(config.dataset.path, "proc.json")))
    split_info = json.load(open(os.path.join(config.dataset.path, "split.json")))

    train_pages = {n: all_pages[n] for n in split_info['train']}
    valid_pages = {n: all_pages[n] for n in split_info['valid']}

    train_dataset = DocPages(train_pages, config)
    val_dataset = DocPages(valid_pages, config)

    train_sampler = DistributedGivenIterationSampler(train_dataset, config.run.total_iter, config.run.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=config.run.batch_size, num_workers=config.run.num_workers,
                              pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    train_val_sampler = DistributedSampler(train_dataset, round_up=False)
    train_val_loader = DataLoader(train_dataset, batch_size=1, num_workers=config.run.num_workers,
                                  pin_memory=True, sampler=train_val_sampler, collate_fn=collate_fn)
    val_sampler = DistributedSampler(val_dataset, round_up=False)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=config.run.num_workers,
                            pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)
    return train_loader, train_val_loader, val_loader
