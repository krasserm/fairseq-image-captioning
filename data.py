import os
import numpy as np
import pandas as pd
import torch

from fairseq.data import FairseqDataset, data_utils
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from PIL import Image


def split_file(split):
    return os.path.join('splits', f'karpathy_{split}_images.txt')


def read_split_image_ids_and_paths(split):
    split_df = pd.read_csv(split_file(split), sep=' ', header=None)
    return split_df.iloc[:, 1].to_numpy(), split_df.iloc[:, 0].to_numpy()


def read_split_image_ids_and_paths_dict(split):
    image_ids_path_dict = {}
    image_ids, image_paths = read_split_image_ids_and_paths(split)
    for image_id, image_path in zip(image_ids, image_paths):
        image_ids_path_dict[image_id] = image_path
    return image_ids_path_dict


def read_split_image_ids(split):
    return read_split_image_ids_and_paths(split)[0]


def read_image_ids(file):
    with open(file, 'r') as f:
        return [int(line) for line in f]


def default_transform():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, image_paths, transform=default_transform()):
        self.image_ids = image_ids
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path).convert('RGB') as img:
            return self.transform(img), self.image_ids[idx]


class ImageCaptionDataset(FairseqDataset):
    def __init__(self, img_ds, cap_ds, cap_dict, shuffle=False):
        self.img_ds = img_ds
        self.cap_ds = cap_ds
        self.cap_dict = cap_dict
        self.shuffle = shuffle

    def __getitem__(self, index):
        source, _ = self.img_ds[index]
        target = self.cap_ds[index]

        return {
            'id': index,
            'source': source,
            'target': target
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

        # Inspired by LanguagePairDataset.ordered_indices
        return indices[np.argsort(self.cap_ds.sizes[indices], kind='mergesort')]

    def collater(self, samples):
        indices = []

        source_samples = []
        target_samples = []
        target_ntokens = 0

        for sample in samples:
            index = sample['id']
            indices.append(index)

            source_samples.append(sample['source'])
            target_samples.append(sample['target'])
            target_ntokens += self.cap_ds.sizes[index]

        num_sentences = len(samples)

        # FIXME: workaround for edge case in parallel processing
        # (framework passes empty samples list
        # to collater under certain conditions)
        if num_sentences == 0:
            return None

        indices = torch.tensor(indices, dtype=torch.long)
        source_batch = default_collate(source_samples)
        target_batch = data_utils.collate_tokens(target_samples, pad_idx=self.cap_dict.pad(), eos_idx=self.cap_dict.eos(), move_eos_to_beginning=False)
        rotate_batch = data_utils.collate_tokens(target_samples, pad_idx=self.cap_dict.pad(), eos_idx=self.cap_dict.eos(), move_eos_to_beginning=True)

        return {
            'id': indices,
            'net_input': {
                'source': source_batch,
                'prev_output_tokens': rotate_batch,
            },
            'target': target_batch,
            'ntokens': target_ntokens,
            'nsentences': num_sentences,
        }
