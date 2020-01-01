#!/bin/bash -e

export PYTHONPATH=./external/coco-caption

python score.py "$@"
