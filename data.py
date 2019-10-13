import os
import glob
import torch
import numpy as np

from fairseq.data import FairseqDataset, data_utils
from torch.utils.data.dataloader import default_collate

from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, transform=lambda x: x):
        self.image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path).convert('RGB') as img:
            return self.transform(img), self.image_id(image_path)

    @staticmethod
    def image_id(image_path):
        image_file = os.path.split(image_path)[1]
        image_name = os.path.splitext(image_file)[0]
        return int(image_name)


class FeaturesDataset(FairseqDataset):
    def __init__(self, features_dir, image_ids_file):
        self.features_dir = features_dir

        with open(image_ids_file, 'r') as f:
            self.image_ids = f.read().splitlines()

    def __getitem__(self, index):
        features_file = os.path.join(self.features_dir, f'{self.image_ids[index]}.npy')
        features = np.load(features_file)

        return torch.as_tensor(features)

    def __len__(self):
        return len(self.image_ids)

    def num_tokens(self, index):
        # TODO: support variable number of features
        return 64

    def size(self, index):
        # TODO: support variable number of features
        return 64

    def collater(self, samples):
        return default_collate(samples)


class ImageCaptionDataset(FairseqDataset):
    def __init__(self, img_ds, cap_ds, cap_dict, shuffle=False):
        self.img_ds = img_ds
        self.cap_ds = cap_ds
        self.cap_dict = cap_dict
        self.shuffle = shuffle

    def __getitem__(self, index):
        img_item = self.img_ds[index]
        cap_item = self.cap_ds[index]

        return {
            'id': index,
            'source': img_item,
            'target': cap_item
        }

    def __len__(self):
        return len(self.cap_ds)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.cap_ds.sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))

        return indices[np.argsort(self.cap_ds.sizes[indices], kind='mergesort')]

    def collater(self, samples):
        ids = [sample['id'] for sample in samples]
        ids = torch.tensor(ids, dtype=torch.long)

        # FIXME: workaround for edge case in parallel processing
        # (framework passes empty samples list
        # to collater under certain conditions)
        if len(samples) == 0:
            return None

        sources = [sample['source'] for sample in samples]
        targets = [sample['target'] for sample in samples]

        num_sentences = len(samples)
        num_tokens = sum([self.cap_ds.sizes[id] for id in ids])

        img_items = self.img_ds.collater(sources)
        img_lengths = torch.full((num_sentences,), fill_value=64, dtype=torch.int64)

        txt_items = data_utils.collate_tokens(targets, pad_idx=self.cap_dict.pad(), eos_idx=self.cap_dict.eos(), move_eos_to_beginning=False)
        prv_items = data_utils.collate_tokens(targets, pad_idx=self.cap_dict.pad(), eos_idx=self.cap_dict.eos(), move_eos_to_beginning=True)


        return {
            'id': ids,
            'net_input': {
                'src_tokens': img_items,
                'src_lengths': img_lengths,
                'prev_output_tokens': prv_items,
            },
            'target': txt_items,
            'ntokens': num_tokens,
            'nsentences': num_sentences,
        }
