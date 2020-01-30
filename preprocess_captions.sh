#!/bin/bash -e

export PYTHONPATH=.:./external/coco-caption

preprocess/preprocess_captions.sh --ms-coco-dir $1 --split train
preprocess/preprocess_captions.sh --ms-coco-dir $1 --split valid
preprocess/preprocess_captions.sh --ms-coco-dir $1 --split test

# Separately tokenize captions for self-critical sequence training
python preprocess/tokenize_captions_scst.py --ms-coco-dir $1
