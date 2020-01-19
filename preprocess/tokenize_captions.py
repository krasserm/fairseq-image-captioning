import argparse
import data
import json
import os
import tqdm

from sacremoses import MosesTokenizer


def load_annotations(coco_dir):
    with open(os.path.join(coco_dir, 'annotations', f'captions_train2014.json')) as f:
        annotations = json.load(f)['annotations']

    with open(os.path.join(coco_dir, 'annotations', f'captions_val2014.json')) as f:
        annotations.extend(json.load(f)['annotations'])

    return annotations


def select_captions(annotations, image_ids):
    """Select captions of given image_ids and return them with their image IDs.
    """

    # for fast lookup
    image_ids = set(image_ids)

    captions = []
    caption_image_ids = []

    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id in image_ids:
            captions.append(annotation['caption'].replace('\n', ''))
            caption_image_ids.append(image_id)

    return captions, caption_image_ids


def tokenize_captions(captions, lang='en'):
    """Tokenizes captions list with Moses tokenizer.
    """

    tokenizer = MosesTokenizer(lang=lang)
    return [tokenizer.tokenize(caption, return_str=True) for caption in captions]


def write_captions(captions, filename, lowercase=True):
    with open(filename, 'w') as f:
        for caption in captions:
            if lowercase:
                caption = caption.lower()
            f.write(caption + '\n')


def write_image_ids(image_ids, filename):
    with open(filename, 'w') as f:
        for image_id in image_ids:
            f.write(f'{image_id}\n')


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load annotations of MS-COCO training and validation set
    annotations = load_annotations(args.ms_coco_dir)

    # Read image ids of given split
    image_ids = data.read_split_image_ids(args.split)

    # Select captions and their image IDs from annotations
    captions, caption_image_ids = select_captions(annotations, image_ids)

    print('Tokenize captions ...')
    captions = tokenize_captions(tqdm.tqdm(captions))

    captions_filename = os.path.join(args.output_dir, f'{args.split}-captions.tok.en')
    caption_image_ids_filename = os.path.join(args.output_dir, f'{args.split}-ids.txt')

    write_captions(captions, captions_filename)
    print(f'Wrote tokenized captions to {captions_filename}.')

    write_image_ids(caption_image_ids, caption_image_ids_filename)
    print(f'Wrote caption image IDs to {caption_image_ids_filename}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-COCO captions pre-processing.')

    parser.add_argument('--ms-coco-dir',
                        help='MS-COCO data directory.')
    parser.add_argument('--split', choices=['train', 'valid', 'test'],
                        help="Data split ('train', 'valid' or 'test').")
    parser.add_argument('--output-dir', default='output',
                        help='Output directory.')

    main(parser.parse_args())

