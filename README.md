### graphsage-age-prediction

Simplified version of https://github.com/bkj/pytorch-graphsage w/ specific application to age prediction in the POKEC benchmark dataset, per [1]

#### Installation

Install `pytorch==0.2.0`, per instructions at: http://pytorch.org/

Then do

```
pip install -r requirements.txt
```

(Exact versions of many of these modules may not actually matter.)

#### Usage

```

# Download datasets
$ ./download.sh

# Prep datasets
$ python prep.py

# Train model
$ python ./train.py --problem-path ./data/pokec/problem.h5

{'epoch': 0, 'train_metric': 3.9663644, 'val_metric': 3.930341}
{'epoch': 1, 'train_metric': 3.3253829, 'val_metric': 3.7660761}
{'epoch': 2, 'train_metric': 2.9626684, 'val_metric': 3.7319703}
```

#### LICENSE
MIT

#### References

[1] http://perozzi.net/publications/15_www_age.pdf
