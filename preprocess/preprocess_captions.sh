#!/bin/bash -e

# Default output dir
OUT_DIR="output"

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --output-dir)
    OUT_DIR="$2"
    shift
    shift
    ;;
    --split)
    SPLIT="$2"
    shift
    shift
    ;;
    *)
    ARGS="$ARGS $1"
    shift
    ;;
esac
done

TOK_PREF="$OUT_DIR"/"${SPLIT}-captions.tok"
TOK_FILE="${TOK_PREF}.en"

BPE_PREF="$OUT_DIR"/"${SPLIT}-captions.bpe"
BPE_FILE="${BPE_PREF}.en"

CODE_FILE="$OUT_DIR"/codes.txt

# Tokenize captions (results are written by script to $TOK_FILE)
python preprocess/tokenize_captions.py --output-dir "$OUT_DIR" --split "$SPLIT" $ARGS

if [[ $SPLIT = "train" ]]
then
    echo "Learn BPE from $TOK_FILE ..."
    subword-nmt learn-bpe -s 10000 < $TOK_FILE > $CODE_FILE
fi

echo "Apply BPE to $TOK_FILE ..."
subword-nmt apply-bpe -c $CODE_FILE < $TOK_FILE > $BPE_FILE

# fairseq-preprocess uses "translation" task by default
# TODO: investigate if it makes sense to use custom tasks

if [[ $SPLIT = "train" ]]
then
    rm -f $OUT_DIR/dict.en.txt

    echo "Generate vocabulary and train dataset files ..."
    # TODO: consider using --nwordssrc option
    fairseq-preprocess --source-lang en --only-source --trainpref $BPE_PREF --destdir $OUT_DIR --thresholdsrc 0

    mv $OUT_DIR/train.en-None.en.bin $OUT_DIR/train-captions.en.bin
    mv $OUT_DIR/train.en-None.en.idx $OUT_DIR/train-captions.en.idx
else
    echo "Generate $SPLIT dataset files ..."
    fairseq-preprocess --source-lang en --only-source --${SPLIT}pref $BPE_PREF --destdir $OUT_DIR --srcdict $OUT_DIR/dict.en.txt

    mv $OUT_DIR/${SPLIT}.en-None.en.bin $OUT_DIR/${SPLIT}-captions.en.bin
    mv $OUT_DIR/${SPLIT}.en-None.en.idx $OUT_DIR/${SPLIT}-captions.en.idx
fi

echo "Done."
