import argparse
import data
import json
import os

from pycocotools.coco import COCO
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def tokenize_captions(output_dir, split, coco):
    image_ids = data.read_image_ids(os.path.join(output_dir, f'{split}-ids.txt'), non_redundant=True)

    gts = dict()

    for image_id in image_ids:
        caps = coco.imgToAnns[image_id]
        gts[image_id] = caps

    return PTBTokenizer().tokenize(gts)


def load_captions_merge(coco_dir):
    with open(os.path.join(coco_dir, 'annotations', f'captions_train2014.json')) as f:
        captions = json.load(f)

    with open(os.path.join(coco_dir, 'annotations', f'captions_val2014.json')) as f:
        captions_val = json.load(f)

    captions['type'] = 'captions'
    captions['images'].extend(captions_val['images'])
    captions['annotations'].extend(captions_val['annotations'])

    return captions


def write_captions(captions, file):
    with open(file, 'w') as f:
        json.dump(captions, f)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    captions = load_captions_merge(args.ms_coco_dir)
    captions_file = os.path.join(args.output_dir, 'all-captions.json')

    write_captions(captions, captions_file)
    print(f'Wrote all captions to {captions_file}')

    coco = COCO(captions_file)

    for split in ['train', 'valid', 'test']:
        captions_tok = tokenize_captions(args.output_dir, split, coco)
        captions_tok_file = os.path.join(args.output_dir, f'{split}-captions.tok.json')

        write_captions(captions_tok, captions_tok_file)
        print(f'Wrote PTB-tokenized {split} captions to {captions_tok_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-COCO captions merging.')

    parser.add_argument('--ms-coco-dir',
                        help='MS-COCO data directory.')
    parser.add_argument('--output-dir', default='output',
                        help='Output directory.')

    main(parser.parse_args())

