#!/bin/bash

# download.sh

mkdir -p ./data/pokec

cd ./data/pokec
wget https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz
wget https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz

gunzip soc-pokec-relationships.txt.gz
gunzip soc-pokec-profiles.txt.gz

cat soc-pokec-profiles.txt  | cut -d$'\t' -f1,8 | sort -k1 -n > soc-pokec-ages.tsv

