import os
import data

from fairseq.data import Dictionary, data_utils
from fairseq.tasks import FairseqTask, register_task

# Import for registration of captioning model
# and architecture at fairseq registry.
import model


@register_task('captioning')
class CaptioningTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--features-dir', default='output',
                            help='image features directory')
        parser.add_argument('--captions-dir', default='output',
                            help='image captions directory')
        parser.add_argument('--captions-lang', default='en', choices=['en'],
                            help='caption language')
        parser.add_argument('--max-source-positions', default=64, type=int, metavar='N',
                            help='max number of image features')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')

    @classmethod
    def setup_task(cls, args, **kwargs):
        captions_dict_file = os.path.join(args.captions_dir, f'dict.{args.captions_lang}.txt')
        captions_dict = Dictionary.load(captions_dict_file)

        return CaptioningTask(args, captions_dict)

    def __init__(self, args, captions_dict):
        super().__init__(args)
        self.captions_dict = captions_dict

    def load_dataset(self, split, **kwargs):
        features_dir = os.path.join(self.args.features_dir, f'{split}-features')
        captions_file = os.path.join(self.args.captions_dir, f'{split}-captions.{self.args.captions_lang}')
        image_ids_file = os.path.join(self.args.captions_dir, f'{split}-ids.txt')

        features_ds = data.FeaturesDataset(features_dir, image_ids_file)
        captions_ds = data_utils.load_indexed_dataset(captions_file, self.captions_dict)

        self.datasets[split] = data.ImageCaptionDataset(features_ds, captions_ds, self.captions_dict, shuffle=True)

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return self.captions_dict
