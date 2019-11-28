#!/bin/bash -e

export PYTHONPATH=.

python preprocess/preprocess_images.py --ms-coco-dir $1 --split train --num-workers 2
python preprocess/preprocess_images.py --ms-coco-dir $1 --split valid --num-workers 2
python preprocess/preprocess_images.py --ms-coco-dir $1 --split test --num-workers 2
