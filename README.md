# Image Captioning Transformer

## Setup

### Environment

- Install [NCCL](https://github.com/NVIDIA/nccl) for multi-GPU training.
- Install [apex](https://github.com/NVIDIA/apex) with the `--cuda_ext` option for faster training.
- Create a conda environment with `conda env create -f environment.yml`.
- Activate the conda environment with `conda activate fairseq-image-captioning`.

### Dataset

Create an `ms-coco` directory in the project's root directory with

    mkdir ms-coco

Download the MS-COCO 2017

- [training images](http://images.cocodataset.org/zips/train2017.zip)
- [validation images](http://images.cocodataset.org/zips/val2017.zip)
- [annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

to the created `ms-coco` directory and extract the archives there. The resulting directory structure should look like

```
ms-coco
  annotations
  images
    train2017
    val2017
```

## Pre-processing

The following command converts MS-COCO images and captions to formats required for model training.

```
export PYTHONPATH=.

./preprocess.sh ms-coco
```

The argument to the `preprocess.sh` script is the path to the MS-COCO directory. Results are stored in the 
created `output` directory. Alternatively, you can also run the individual pre-processing steps explicitly with   

```
export PYTHONPATH=.

./preprocess_captions.sh --ms-coco-dir ms-coco --split train
./preprocess_captions.sh --ms-coco-dir ms-coco --split valid

./preprocess_images.sh --ms-coco-dir ms-coco --split train --num-workers 2
./preprocess_images.sh --ms-coco-dir ms-coco --split valid --num-workers 2
```

## Training

Model training can be started with 

```
python -m fairseq_cli.train \
    --user-dir task \
    --task captioning \
    --arch captioning-transformer \
    --features-dir output \
    --captions-dir output \
    --save-dir .checkpoints \
    --criterion cross_entropy \
    --optimizer nag \
    --lr 0.001 \
    --encoder-embed-dim 512 \
    --max-tokens 1024 \
    --num-workers 2 \
    --log-interval 10 \
    --save-interval-updates 1000 \
    --keep-interval-updates 3 \
    --max-epoch 25 \
    --no-epoch-checkpoints \
    --no-progress-bar \
    --simplistic-encoder
```

The `--simplistic-encoder` option uses an encoder that only projects the 2048-dimensional image features to the 
512-dimensional encoder embedding space. If this option is omitted, then two additional transformer encoder layers
are used to process the projected features (without using positional encodings at the moment).
  
## Evaluation

Training writes checkpoints every 1000 updates. To produce captions for some [sample validation images](eval-images.txt) 
using the best checkpoint so far run

``` 
python eval.py \
    --user-dir task \
    --features-dir output \
    --captions-dir output \
    --path .checkpoints/checkpoint_last.pt \
    --tokenizer moses \
    --bpe subword_nmt \
    --bpe-codes output/codes.txt \
    --beam 5 \
    --input eval-images.txt
``` 

The corresponding image files are located in the `ms-coco/images/val2017` directory. After a few training epochs you 
should be able to see some reasonable caption generated, good captions after approx. 15-20 epochs. You can run 
`eval.py` while training is in progress.
