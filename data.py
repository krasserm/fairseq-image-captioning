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

        self.sizes = np.ones(len(self.image_ids), dtype=np.int) * 64

    def __getitem__(self, index):
        features_file = os.path.join(self.features_dir, f'{self.image_ids[index]}.npy')
        features = np.load(features_file)
        features = features[:self.num_tokens(index)]
        return torch.as_tensor(features)

    def __len__(self):
        return len(self.image_ids)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

    def collater(self, samples):
        num_tokens = [sample.shape[0] for sample in samples]
        max_tokens = max(num_tokens)

        samples_padded = []

        for s, n in zip(samples, num_tokens):
            sample_padded = np.pad(s, pad_width=((0, max_tokens-n), (0, 0)), mode='constant')
            samples_padded.append(sample_padded)

        return default_collate(samples_padded)


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
        return self.size(index)[1]

    def size(self, index):
        # number of image feature vectors, number of tokens in caption
        return self.img_ds.sizes[index], self.cap_ds.sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))

        # Inspired by LanguagePairDataset.ordered_indices
        indices = indices[np.argsort(self.cap_ds.sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.img_ds.sizes[indices], kind='mergesort')]

    def collater(self, samples):
        ids = [sample['id'] for sample in samples]
        ids = torch.tensor(ids, dtype=torch.long)

        num_sentences = len(samples)

        # FIXME: workaround for edge case in parallel processing
        # (framework passes empty samples list
        # to collater under certain conditions)
        if num_sentences == 0:
            return None

        sources = [sample['source'] for sample in samples]
        targets = [sample['target'] for sample in samples]

        img_lengths = torch.tensor([self.img_ds.size(id) for id in ids])
        img_items = self.img_ds.collater(sources)

        txt_lengths = sum([self.cap_ds.sizes[id] for id in ids])
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
            'ntokens': txt_lengths,
            'nsentences': num_sentences,
        }
