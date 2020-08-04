#!/bin/bash -e

export PYTHONPATH=.:./external/coco-caption

for split in 'train' 'valid' 'test'
do
  preprocess/preprocess_captions.sh --ms-coco-dir $1 --split $split
done

# Separately tokenize captions for self-critical sequence training
python preprocess/tokenize_captions_scst.py --ms-coco-dir $1
