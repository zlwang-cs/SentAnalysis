import os

dataset = 'IMDB'
raw_dataset_path = "/home/wangzilong/Documents/Code/SentAnalysis/data/raw/aclImdb"
proc_dataset_path = f"/home/wangzilong/Documents/Code/SentAnalysis/data/{dataset}"

if not os.path.exists(proc_dataset_path):
    os.mkdir(proc_dataset_path)


def proc_dataset(name):
    with open(os.path.join(proc_dataset_path, name+'.txt'), 'w') as fout:
        path = os.path.join(raw_dataset_path, name, 'pos')
        files = os.listdir(path)
        for f in files:
            text = open(os.path.join(path, f)).readlines()
            assert len(text) == 1
            text = text[0]
            print(f"{text}\t{1}", file=fout)

        path = os.path.join(raw_dataset_path, name, 'neg')
        files = os.listdir(path)
        for f in files:
            text = open(os.path.join(path, f)).readlines()
            assert len(text) == 1
            text = text[0]
            print(f"{text}\t{0}", file=fout)


proc_dataset('train')
proc_dataset('test')

