#!/bin/bash -e

export PYTHONPATH=.

for split in 'train' 'valid' 'test'
do
  python preprocess/preprocess_images.py --ms-coco-dir $1 --split $split --num-workers 2
done
