# Image Captioning Transformer

This project implements [Transformer](https://arxiv.org/abs/1706.03762)\-based image captioning models as extension of 
the [pytorch/fairseq](https://github.com/pytorch/fairseq) sequence modeling toolkit. It is still in an early stage, 
don't use it for any serious work. The current implementation is based on ideas from the following papers:

- Jun Yu, Jing Li, Zhou Yu, and Qingming Huang. [Multimodal transformer with multi-view visual
  representation for image captioning](https://arxiv.org/abs/1905.07841). arXiv preprint arXiv:1905.07841, 2019.
  
- Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. [Conceptual captions: A cleaned, hypernymed, image 
  alt-text dataset for automatic image captioning](https://www.aclweb.org/anthology/P18-1238/). In ACL, 2018.

- P. Anderson, X. He, C. Buehler, D. Teney, M. Johnson, S. Gould, and L. Zhang. [Bottom-up and top-down
  attention for image captioning and visual question answering](https://arxiv.org/abs/1707.07998). 
  In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6077–6086, 2018.
  
- Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhudinov, Rich Zemel, and Yoshua
  Bengio. [Show, Attend and Tell: Neural image caption generation with visual attention](https://arxiv.org/abs/1502.03044). 
  In International Conference on Machine Learning. 2015.
   
## Setup

### Environment

- Install [NCCL](https://github.com/NVIDIA/nccl) for multi-GPU training.
- Install [apex](https://github.com/NVIDIA/apex) with the `--cuda_ext` option for faster training.
- Create a conda environment with `conda env create -f environment.yml`.
- Activate the conda environment with `conda activate fairseq-image-captioning`.

### Dataset

Models are currently trained with the MS-COCO dataset. To setup the dataset for training, create an `ms-coco` directory 
in the project's root directory, download MS-COCO 2014

- [training images](http://images.cocodataset.org/zips/train2014.zip) (13 GB)
- [validation images](http://images.cocodataset.org/zips/val2014.zip) (6 GB)
- [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) (241 MB)

to the created `ms-coco` directory and extract the archives there. The resulting directory structure should look like

```
ms-coco
  annotations
  images
    train2014
    val2014
```

MS-COCO images are needed when training with the `--features grid` command line option. Image features are then extracted
from a fixed 8 x 8 grid on the image. When using the `--features obj` command line option image features are extracted 
from detected objects (see also *bottom-up attention* in [this paper](https://arxiv.org/abs/1707.07998)).
 
Pre-computed features of detected objects (10-100 per image) are available in [this repository](https://github.com/peteanderson80/bottom-up-attention). 
You can also use [this link](https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip) for downloading them 
directly (22 GB). After downloading, extract the `trainval.zip` file, rename the `trainval` directory to `features` and 
move it to the `ms-coco` directory. The `ms-coco/features` directory should contain 4 `.tsv` files.

```
ms-coco
  annotations
  features
    karpathy_test_resnet101_faster_rcnn_genome.tsv
    karpathy_train_resnet101_faster_rcnn_genome.tsv.0
    karpathy_train_resnet101_faster_rcnn_genome.tsv.1
    karpathy_val_resnet101_faster_rcnn_genome.tsv
  images
    train2014
    val2014
```

## Pre-processing

For splitting the downloaded MS-COCO data into a training, validation and test set, [Karpathy splits](splits) are used. 
Split files have been copied from [this location](https://github.com/peteanderson80/bottom-up-attention/tree/master/data/genome/coco_splits).
All pre-processing commands in the following sub-sections write their results to the `output` directory.  

### Pre-process captions

    ./preprocess_captions.sh ms-coco

Converts MS-COCO captions into a format required for model training. 

### Pre-process images

    ./preprocess_images.sh ms-coco

Converts MS-COCO images into a format required for model training. Only needed when training with the `--features grid`
command line option.

### Pre-process object features

    ./preprocess_features.sh ms-coco/features

Converts pre-computed object features into a format required for model training. Only needed when training with the 
`--features obj` command line option.

## Extensions

In addition to all [fairseq command line options](https://fairseq.readthedocs.io/en/latest/command_line_tools.html) this
project implements the following extensions:

- `--task captioning`. Enables the image captioning functionality implemented by this project. 
- `--features grid`. Use image features extracted from an 8 x 8 grid. Inception v3 is used for extracting image features. 
  Additionally use `--max-source-positions 64` when using this option. 
- `--features obj`. Use image features extracted from detected objects as described in [this paper](https://arxiv.org/abs/1707.07998).
  Additionally use `--max-source-positions 100` when using this option.
- `--arch default-captioning-arch`. Uses a transformer encoder to process image features (2 layers by default) and a 
  transformer decoder to process image captions and encoder output (6 layers by default). The number of encoder and 
  decoder layers can be adjusted with `--encoder-layers` and `--decoder-layers`, respectively.
- `--arch simplistic-captioning-arch`. Uses the same decoder as in `default-captioning-arch` but no transformer encoder.
  Image features are processed directly by the decoder after projecting them into a lower-dimensional space which can
  be controlled with `--encoder-embed-dim`. Projection into lower-dimensional space can be skipped with `--no-projection`.
- `--feature-spatial-embeddings`. Learns positional (spatial) embeddings of bounding boxes or grid tiles. Disabled by default. 
  Positional embeddings are learned from the top-left and bottom-right coordinates of boxes/tiles and their relative sizes.
  
## Training

An example command for training a simple captioning model is:  

```
python -m fairseq_cli.train \
       --task captioning \
       --arch simplistic-captioning-arch \
       --features grid \
       --features-dir output \
       --captions-dir output \
       --user-dir task \
       --save-dir .checkpoints \
       --optimizer nag \
       --lr 0.001 \
       --criterion cross_entropy \
       --max-epoch 50 \
       --max-tokens 1024 \
       --max-source-positions 64 \
       --encoder-embed-dim 512 \
       --log-interval 10 \
       --save-interval-updates 1000 \
       --keep-interval-updates 3 \
       --num-workers 2 \
       --no-epoch-checkpoints \
       --no-progress-bar
```

See [Extensions](#extensions) for captioning-specific command line options. Checkpoints are written to a `.checkpoints` 
directory and `.checkpoints/checkpoint_best.pt` should be used for testing. **Please note that the hyper-parameters used 
here are just examples, they are not tuned yet**. 

## Demo

Scripts for detailed model evaluation are not available yet but will come soon, together with an application that reads
images from a directory to caption them.  At the moment you can use the following simple demo application for captioning 
images of the validation dataset. 

```
python demo.py \
       --features grid \
       --features-dir output \
       --captions-dir output \
       --user-dir task \
       --tokenizer moses \
       --bpe subword_nmt \
        --bpe-codes output/codes.txt \
       --beam 5 \
       --path .checkpoints/checkpoint_best.pt \
       --input demo/val-images.txt
```



Validation image IDs are read from [demo/val-images.txt](demo/val-images.txt). This should produce an output containing 
something like 

```
105537: A street sign hanging from the side of a metal pole.
130599: A man standing next to a giraffe statue.
...
```

### Pre-trained model

A model obtained with the [training](#training) command above is available for download ([checkpoint_demo.pt](https://drive.google.com/open?id=1GWLenxZivitAcniSUXRcaC8N8b5_7two)).
Assuming you've downloaded the file to the project's root directory you can run the demo with 

```
python demo.py \
       --features grid \
       --features-dir output \
       --captions-dir output \
       --user-dir task \
       --tokenizer moses \
       --bpe subword_nmt \
        --bpe-codes output/codes.txt \
       --beam 5 \
       --path checkpoint_demo.pt \
       --input demo/val-images.txt
```

Two sample validation images and their produced captions are:

![130599](demo/COCO_val2014_000000130599.jpg)

"A man standing next to a giraffe statue."

![105537](demo/COCO_val2014_000000105537.jpg)

"A street sign hanging from the side of a metal pole."


