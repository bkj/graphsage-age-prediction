### graphsage-age-prediction

Simplified version of https://github.com/bkj/pytorch-graphsage w/ specific application to age prediction in the POKEC benchmark dataset, per [1]

#### Installation

Install `pytorch==0.2.0`, per instructions at: http://pytorch.org/

#### Usage

```

# Download datasets
./download.sh

# Prep datasets
python prep.py

# Train model
python ./train.py --problem-path ./data/pokec/problem.h5

```

#### LICENSE
MIT

#### References

[1] http://perozzi.net/publications/15_www_age.pdf