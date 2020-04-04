# SentAnalysis

**Zilong Wang**

An easy project about textual sentiment analysis.

## Dataset:
**IMDB:** 
 - put the original dataset in `data`. 
 - Run `python data/processing/create_dataset.py` to create dataset for training and testing.

## Train:
 - Customize the config in `config/default.yaml`.
 - Run `python main.py`.

## Test:
 - The running log is in `log`.
 - Run `python test.py --dir $DIR --weight $MODEL`
