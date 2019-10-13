import argparse
import json
import os
import tqdm

from sacremoses import MosesTokenizer


def load_annotations(coco_dir, coco_split):
    """Loads MS-COCO captions JSON and returns value of 'annotations' key.

       Args:
           coco_dir (str): MS-COCO data directory
           coco_split (str): 'train2017' or 'val2017'
    """

    with open(os.path.join(coco_dir, 'annotations', f'captions_{coco_split}.json')) as f:
        return json.load(f)['annotations']


def get_captions_and_image_ids(annotations):
    """Extracts captions and image IDs from annotations and return them as lists.
    """

    image_captions = []
    image_ids = []

    for annotation in annotations:
        image_captions.append(annotation['caption'].replace('\n', ''))
        image_ids.append(annotation['image_id'])

    return image_captions, image_ids


def tokenize_captions(captions, lang='en'):
    """Tokenizes captions list with Moses tokenizer.
    """

    tokenizer = MosesTokenizer(lang=lang)
    return [tokenizer.tokenize(caption, return_str=True) for caption in captions]


def write_captions(captions, filename):
    with open(filename, 'w') as f:
        for caption in captions:
            f.write(caption + '\n')


def write_image_ids(image_ids, filename):
    with open(filename, 'w') as f:
        for image_id in image_ids:
            f.write(f'{image_id}\n')


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    splits = {
        'train': args.ms_coco_train_split,
        'valid': args.ms_coco_valid_split
    }

    # Load annotations from MS-COCO dataset
    annotations = load_annotations(args.ms_coco_dir, splits[args.split])

    # Extract captions and image IDs from annotations
    captions, ids = get_captions_and_image_ids(annotations)

    print('Tokenize captions ...')
    captions = tokenize_captions(tqdm.tqdm(captions))

    captions_filename = os.path.join(args.output_dir, f'{args.split}-captions.tok.en')
    ids_filename = os.path.join(args.output_dir, f'{args.split}-ids.txt')

    write_captions(captions, captions_filename)
    print(f'Wrote tokenized captions to {captions_filename}.')

    write_image_ids(ids, ids_filename)
    print(f'Wrote image IDs to {ids_filename}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-COCO captions pre-processing.')

    parser.add_argument('--ms-coco-dir',
                        help='MS-COCO data directory.')
    parser.add_argument('--ms-coco-train-split', default='train2017',
                        help='MS-COCO training data split name.')
    parser.add_argument('--ms-coco-valid-split', default='val2017',
                        help='MS-COCO validation data split name.')
    parser.add_argument('--split', choices=['train', 'valid'],
                        help="Data split ('train' or 'valid').")
    parser.add_argument('--output-dir', default='output',
                        help='Output directory.')

    main(parser.parse_args())

