# Image Captioning Transformer

This project implements [Transformer](https://arxiv.org/abs/1706.03762)\-based image captioning models as extension of 
the [pytorch/fairseq](https://github.com/pytorch/fairseq) sequence modeling toolkit. It is still in an early stage, 
don't use it for any serious work. The current implementation is based on ideas from the following papers:

- Jun Yu, Jing Li, Zhou Yu, and Qingming Huang. [Multimodal transformer with multi-view visual
  representation for image captioning](https://arxiv.org/abs/1905.07841). arXiv preprint arXiv:1905.07841, 2019.
  
- X. Zhu, L. Li, L. Jing, H. Peng, X. Niu. [Captioning Transformer with Stacked Attention Modules](https://www.mdpi.com/2076-3417/8/5/739). 
  Appl. Sci. 2018, 8, 739

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
- `--feature-spatial-encoding`. Learns spatial (positional) encodings of bounding boxes or grid tiles. Disabled by default. 
  Positional encodings are learned of the top-left and bottom-right coordinates of boxes/tiles and their relative sizes.
  
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

## Inference

Inference can be performed using the `inference.py` script to generate captions for images of a specified dataset.    
A saved model, a file containing image IDs and the dataset-split these image IDs are taken from have to be supplied.
An optional output path may be specified to store the predicted captions as a json-file.

The following command shows an example usage of the script:

```
python inference.py \
       --features grid \
       --features-dir output \
       --captions-dir output \
       --user-dir task \
       --tokenizer moses \
       --bpe subword_nmt \
       --bpe-codes output/codes.txt \
       --beam 5 \
       --path .checkpoints/checkpoint_best.pt \
       --split test \
       --input demo/test-images.txt \
       --output demo/test-predictions.json
```

Image IDs are read from [demo/test-images.txt](demo/test-images.txt) in this sample. 
This should produce an output containing something like:

```
314251: A group of people riding bikes down a street.
208408: A black and white photo of a person walking on a sidewalk.
...
```

Furthermore, the predictions are stored in [demo/test-predictions.json](demo/test-predictions.json).

## Evaluation

### Installation of dependencies

Evaluation is performed using a [Python 3 Fork](https://github.com/flauted/coco-caption) of the
[COCO Caption Evaluation](https://github.com/tylin/coco-caption) library. This library is included
as a [Git Submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) in the path `external/coco-caption`.

Download the submodule using following commands:

```
git submodule init
git submodule update
``` 

Furthermore, the COCO Caption Evaluation library uses the [Stanford CoreNLP 3.6.0](https://stanfordnlp.github.io/CoreNLP/index.html) toolset
which must be downloaded prior to the first execution of the scoring script. 

It is important to change into the submodules root folder before executing the script to download the required files:

```
cd external/coco-caption
./get_stanford_models.sh
```

### Model evaluation

Models can be evaluated with the `score.sh` script. This script uses the specified reference-captions as the
ground truth and evaluates the model based on the generated captions provided as a json-file (created prior via the 
`inference.py` script). The following example calculates the evaluation scores for the generated captions provided in 
[demo/test-predictions.json](demo/test-predictions.json).

```
./score.sh \
        --reference-captions external/coco-caption/annotations/captions_val2014.json \
        --system-captions demo/test-predictions.json
```

### Evaluating model performance on the test set

To evaluate a trained model on the test set, the following command sequence should be used:

```
# Generate captions for all images in the Karpathy test set
python inference.py \
       --features grid \
       --features-dir output \
       --captions-dir output \
       --user-dir task \
       --tokenizer moses \
       --bpe subword_nmt \
       --bpe-codes output/codes.txt \
       --beam 5 \
       --path .checkpoints/checkpoint_best.pt \
       --split test \
       --input output/test-ids.txt \
       --output output/test-predictions.json

# Score the generated captions
./score.sh \
        --reference-captions external/coco-caption/annotations/captions_val2014.json \
        --system-captions output/test-predictions.json
```

Note that the `test` set used to evaluate the model is taken from the [Karpathy splits](splits) and is a subset of the 
official MS-COCO `val2014` set.

### Pre-trained model

A model obtained with the [training](#training) command above is available for download ([checkpoint_demo.pt](https://drive.google.com/open?id=1GWLenxZivitAcniSUXRcaC8N8b5_7two)).
**It has only been trained for a few epochs, so don't expect high-quality captions.**  Assuming you've downloaded the 
model file to the project's root directory you can run the demo with 

```
python inference.py \
       --features grid \
       --features-dir output \
       --captions-dir output \
       --user-dir task \
       --tokenizer moses \
       --bpe subword_nmt \
       --bpe-codes output/codes.txt \
       --beam 5 \
       --path checkpoint_demo.pt \
       --split test \
       --input demo/test-images.txt
```

Two sample test images and their produced captions are:

![130599](demo/COCO_val2014_000000314251.jpg)

"A group of people riding bikes down a street."

![105537](demo/COCO_val2014_000000208408.jpg)

"A black and white photo of a person walking on a sidewalk."