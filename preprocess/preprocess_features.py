import argparse
import base64
import csv
import os
import sys
import torch
import tqdm
import numpy as np


csv.field_size_limit(sys.maxsize)

FIELDNAMES_IN = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
FIELDNAMES_OUT = FIELDNAMES_IN[:-2]


def features_files(features_dir):
    return {
        'train': [os.path.join(features_dir, f'karpathy_train_resnet101_faster_rcnn_genome.tsv.0'),
                  os.path.join(features_dir, f'karpathy_train_resnet101_faster_rcnn_genome.tsv.1')],
        'valid': [os.path.join(features_dir, f'karpathy_val_resnet101_faster_rcnn_genome.tsv')],
        'test': [os.path.join(features_dir, f'karpathy_test_resnet101_faster_rcnn_genome.tsv')]
    }


def main(args):
    in_features_files = features_files(args.features_dir)[args.split]
    out_features_dir = os.path.join(args.output_dir, f'{args.split}-features-obj')
    out_metadata_file = os.path.join(out_features_dir, 'metadata.csv')

    os.makedirs(out_features_dir, exist_ok=True)

    with open(out_metadata_file, 'w') as fo:
        writer = csv.DictWriter(fo, fieldnames=FIELDNAMES_OUT, extrasaction='ignore')
        writer.writeheader()

        for in_features_file in in_features_files:
            with open(in_features_file, 'r') as fi:
                reader = csv.DictReader(fi, delimiter='\t', fieldnames=FIELDNAMES_IN)
                for item in tqdm.tqdm(reader):
                    item['image_id'] = int(item['image_id'])
                    item['image_h'] = int(item['image_h'])
                    item['image_w'] = int(item['image_w'])
                    item['num_boxes'] = int(item['num_boxes'])
                    for field in ['boxes', 'features']:
                        item[field] = np.frombuffer(base64.decodebytes(item[field].encode()), dtype=np.float32).reshape(item['num_boxes'], -1)

                    # write features metadata
                    # TODO: include bounding boxes
                    writer.writerow(item)

                    # write features, one file per image
                    np.save(os.path.join(out_features_dir, f"{item['image_id']}.npy"), item['features'])

                    # write bounding boxes, one file per image
                    np.save(os.path.join(out_features_dir, f"{item['image_id']}-boxes.npy"), item['boxes'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object features pre-processing.')

    parser.add_argument('--features-dir',
                        help='Object features data directory.')
    parser.add_argument('--split', choices=['train', 'valid', 'test'],
                        help="Data split ('train', 'valid' or 'test').")
    parser.add_argument('--output-dir', default='output',
                        help='Output directory.')
    parser.add_argument('--device', default='cuda', type=torch.device,
                        help="Device to use ('cpu', 'cuda', ...).")

    main(parser.parse_args())
