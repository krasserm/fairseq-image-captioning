#!/bin/bash -e

export PYTHONPATH=.

for split in 'train' 'valid' 'test'
do
  python preprocess/preprocess_features.py --features-dir $1 --split $split
done
