import re
import os
import argparse

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def main(reference_caption_file: str, system_caption_file: str):
    coco = COCO(reference_caption_file)
    coco_system_captions = coco.loadRes(system_caption_file)
    coco_eval = COCOEvalCap(coco, coco_system_captions)
    coco_eval.params['image_id'] = coco_system_captions.getImgIds()

    coco_eval.evaluate()

    print('\nScores:')
    print('=======')
    for metric, score in coco_eval.eval.items():
        print('{}: {:.3f}'.format(metric, score))


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    main(args.reference_captions, args.system_captions)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-captions', type=lambda x: is_valid_file(parser, x, ['json']), required=True)
    parser.add_argument('--system-captions', type=lambda x: is_valid_file(parser, x, ['json']), required=True)
    return parser


def is_valid_file(parser, arg, file_types):
    ext = re.sub(r'^\.', '', os.path.splitext(arg)[1])

    if not os.path.exists(arg):
        parser.error('File not found: "{}"'.format(arg))
    elif ext not in file_types:
        parser.error('Invalid "{}" provided. Only files of type {} are allowed'.format(arg, file_types))
    else:
        return arg


if __name__ == '__main__':
    cli_main()
