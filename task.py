import os
import data

from fairseq.data import Dictionary, data_utils
from fairseq.tasks import FairseqTask, register_task

# Import for registration of captioning model and architecture at fairseq registry.
from model import captioning


@register_task('captioning')
class CaptioningTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--ms-coco-dir',
                            help='MS-COCO data directory.')
        parser.add_argument('--preproc-dir', default='output',
                            help='pre-processing output directory')
        parser.add_argument('--captions-lang', default='en', choices=['en'],
                            help='caption language')
        parser.add_argument('--max-source-positions', default=64, type=int, metavar='N',
                            help='max number of objects in the source image')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')

    @classmethod
    def setup_task(cls, args, **kwargs):
        captions_dict_file = os.path.join(args.preproc_dir, f'dict.{args.captions_lang}.txt')
        captions_dict = Dictionary.load(captions_dict_file)

        return CaptioningTask(args, captions_dict)

    def __init__(self, args, captions_dict):
        super().__init__(args)
        self.captions_dict = captions_dict

    def load_dataset(self, split, **kwargs):
        captions_file = os.path.join(self.args.preproc_dir, f'{split}-captions.{self.args.captions_lang}')
        captions_ds = data_utils.load_indexed_dataset(captions_file, self.captions_dict)

        image_ids_file = os.path.join(self.args.preproc_dir, f'{split}-ids.txt')
        image_ids = data.read_image_ids(image_ids_file)

        image_ids_path_dict = data.read_split_image_ids_and_paths_dict(split)
        image_paths = [os.path.join(self.args.ms_coco_dir, 'images', image_ids_path_dict[image_id]) for image_id in image_ids]
        image_ds = data.ImageDataset(image_ids, image_paths)

        self.datasets[split] = data.ImageCaptionDataset(image_ds, captions_ds, self.captions_dict, shuffle=True)

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return self.captions_dict
