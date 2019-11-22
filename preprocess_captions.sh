#!/bin/bash -e

preprocess/preprocess_captions.sh --ms-coco-dir $1 --split train
preprocess/preprocess_captions.sh --ms-coco-dir $1 --split valid
preprocess/preprocess_captions.sh --ms-coco-dir $1 --split test
