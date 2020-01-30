import os
import data

from fairseq.data import Dictionary, data_utils
from fairseq.tasks import FairseqTask, register_task

# Import for registration of captioning model
# and self-critical sequence training criterion.
import model.caption
import scst.criterion


@register_task('captioning')
class CaptioningTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--features', default='grid', choices=['grid', 'obj'],
                            help='image features')
        parser.add_argument('--features-dir', default='output',
                            help='image features directory')
        parser.add_argument('--captions-dir', default='output',
                            help='image captions directory')
        parser.add_argument('--captions-lang', default='en', choices=['en'],
                            help='caption language')
        parser.add_argument('--max-source-positions', default=64, type=int, metavar='N',
                            help='max number of objects in the source image')
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
        self.scst = args.criterion == 'self_critical_sequence_training'

    def load_dataset(self, split, **kwargs):
        features_dir = os.path.join(self.args.features_dir, f'{split}-features-{self.args.features}')

        image_ids_file = os.path.join(self.args.captions_dir, f'{split}-ids.txt')
        image_ids = data.read_image_ids(image_ids_file, non_redundant=self.scst)

        if self.scst and split == 'valid':
            image_ids = image_ids[:self.args.scst_validation_set_size]

        if self.scst:
            captions_file = os.path.join(self.args.captions_dir, f'{split}-captions.tok.json')
            captions_ds = data.CaptionsDataset(captions_file, image_ids)
        else:
            captions_file = os.path.join(self.args.captions_dir, f'{split}-captions.{self.args.captions_lang}')
            captions_ds = data_utils.load_indexed_dataset(captions_file, self.captions_dict)

        if self.args.features == 'grid':
            image_ds = data.GridFeaturesDataset(features_dir, image_ids, grid_shape=(8, 8))
        elif self.args.features == 'obj':
            image_metadata_file = os.path.join(features_dir, 'metadata.csv')
            image_metadata = data.read_image_metadata(image_metadata_file)
            image_ds = data.ObjectFeaturesDataset(features_dir, image_ids, image_metadata)
        else:
            raise ValueError(f'Invalid --features option: {self.args.features}')

        self.datasets[split] = data.ImageCaptionDataset(img_ds=image_ds,
                                                        cap_ds=captions_ds,
                                                        cap_dict=self.captions_dict,
                                                        scst=self.scst,
                                                        shuffle=True)

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return self.captions_dict
