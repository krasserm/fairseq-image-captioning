#!/bin/bash -e

export PYTHONPATH=.

python preprocess/preprocess_features.py --features-dir $1 --split train
python preprocess/preprocess_features.py --features-dir $1 --split valid
python preprocess/preprocess_features.py --features-dir $1 --split test
