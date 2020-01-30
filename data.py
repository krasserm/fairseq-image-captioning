import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from fairseq.data import FairseqDataset, data_utils
from torch.utils.data.dataloader import default_collate

from PIL import Image


def split_file(split):
    return os.path.join('splits', f'karpathy_{split}_images.txt')


def read_split_image_ids_and_paths(split):
    split_df = pd.read_csv(split_file(split), sep=' ', header=None)
    return split_df.iloc[:,1].to_numpy(), split_df.iloc[:,0].to_numpy()


def read_split_image_ids(split):
    return read_split_image_ids_and_paths(split)[0]


def read_image_ids(file, non_redundant=False):
    with open(file, 'r') as f:
        image_ids = [int(line) for line in f]

    if non_redundant:
        return list(set(image_ids))
    else:
        return image_ids


def read_image_metadata(file):
    df = pd.read_csv(file)
    md = {}

    for img_id, img_h, img_w, num_boxes in zip(df['image_id'], df['image_h'], df['image_w'], df['num_boxes']):
        md[img_id] = {
            'image_h': np.float32(img_h),
            'image_w': np.float32(img_w),
            'num_boxes': num_boxes
        }

    return md


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, image_paths, transform=lambda x: x):
        self.image_ids = image_ids
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path).convert('RGB') as img:
            return self.transform(img), self.image_ids[idx]


class FeaturesDataset(FairseqDataset):
    def __init__(self, features_dir, image_ids, num_objects):
        self.features_dir = features_dir
        self.image_ids = image_ids
        self.num_objects = num_objects

    def __getitem__(self, index):
        return self.read_data(self.image_ids[index])

    def __len__(self):
        return len(self.image_ids)

    def num_tokens(self, index):
        return self.num_objects[index]

    def size(self, index):
        return self.num_objects[index]

    @property
    def sizes(self):
        return self.num_objects

    def read_data(self, image_id):
        raise NotImplementedError

    def collater(self, samples):
        num_objects = [features.shape[0] for features, _ in samples]
        max_objects = max(num_objects)

        feature_samples_padded = []
        location_samples_padded = []

        for (features, locations), n in zip(samples, num_objects):
            features_padded = F.pad(features, pad=[0, 0, 0, max_objects-n], mode='constant', value=0.0)
            locations_padded = F.pad(locations, pad=[0, 0, 0, max_objects-n], mode='constant', value=0.0)
            feature_samples_padded.append(features_padded)
            location_samples_padded.append(locations_padded)

        return default_collate(feature_samples_padded), default_collate(location_samples_padded)


class GridFeaturesDataset(FeaturesDataset):
    def __init__(self, features_dir, image_ids, grid_shape=(8, 8)):
        super().__init__(features_dir=features_dir,
                         image_ids=image_ids,
                         num_objects=np.ones(len(image_ids), dtype=np.int) * np.prod(grid_shape))

        self.grid_shape = grid_shape
        self.locations = self.tile_locations(grid_shape)

    def read_data(self, image_id):
        features_file = os.path.join(self.features_dir, f'{image_id}.npy')
        features = np.load(features_file)
        return torch.as_tensor(features), self.locations

    @staticmethod
    def tile_locations(grid_shape):
        num_tiles = np.prod(grid_shape)
        rel_tile_w = 1. / grid_shape[1]
        rel_tile_h = 1. / grid_shape[0]
        rel_tile_area = 1. / num_tiles

        rel_tile_locations = np.zeros(shape=(grid_shape[0], grid_shape[1], 5), dtype=np.float32)

        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                rel_tile_locations[i, j] = np.array([
                    j * rel_tile_w,
                    i * rel_tile_h,
                    (j+1) * rel_tile_w,
                    (i+1) * rel_tile_h,
                    rel_tile_area
                ], dtype=np.float32)

        return torch.as_tensor(rel_tile_locations).view(num_tiles, 5)


class ObjectFeaturesDataset(FeaturesDataset):
    def __init__(self, features_dir, image_ids, image_metadata):
        super().__init__(features_dir=features_dir,
                         image_ids=image_ids,
                         num_objects=np.array([image_metadata[image_id]['num_boxes'] for image_id in image_ids]))

        self.image_metadata = image_metadata

    def read_data(self, image_id):
        features_file = os.path.join(self.features_dir, f'{image_id}.npy')
        features = np.load(features_file)

        boxes_file = os.path.join(self.features_dir, f'{image_id}-boxes.npy')
        boxes = np.load(boxes_file)

        # Normalize box coordinates
        boxes[:, [0, 2]] /= self.image_metadata[image_id]['image_w']
        boxes[:, [1, 3]] /= self.image_metadata[image_id]['image_h']

        # Normalized box areas
        areas = (boxes[:, 2] - boxes[:, 0]) * \
                (boxes[:, 3] - boxes[:, 1])

        return torch.as_tensor(features), \
               torch.as_tensor(np.c_[boxes, areas])


class CaptionsDataset(FairseqDataset):
    """Captions dataset used for self-critical sequence training (SCST) only.
    """
    def __init__(self, captions_file, image_ids):
        self.image_ids = image_ids
        self.num_objects = np.zeros(len(image_ids), dtype=np.int)

        with open(captions_file) as f:
            self.captions = json.load(f)

        for i, image_id in enumerate(image_ids):
            captions = self.captions[str(image_id)]
            caption_sizes = [len(caption.split(' ')) for caption in captions]
            self.num_objects[i] = np.max(caption_sizes)

    def __getitem__(self, index):
        return self.captions[str(self.image_ids[index])]

    def __len__(self):
        return len(self.image_ids)

    def num_tokens(self, index):
        return self.num_objects[index]

    def size(self, index):
        return self.num_objects[index]

    @property
    def sizes(self):
        return self.num_objects

    def collater(self, samples):
        return samples


class ImageCaptionDataset(FairseqDataset):
    def __init__(self, img_ds, cap_ds, cap_dict, scst=False, shuffle=False):
        self.img_ds = img_ds
        self.cap_ds = cap_ds
        self.cap_dict = cap_dict
        self.scst = scst
        self.shuffle = shuffle

    def __getitem__(self, index):
        source_features, source_locations = self.img_ds[index]
        target = self.cap_ds[index]

        return {
            'id': index,
            'source_features': source_features,
            'source_locations': source_locations,
            'target': target
        }

    def __len__(self):
        return len(self.img_ds)

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
        indices = []

        source_feature_samples = []
        source_location_samples = []
        source_lengths = []

        target_samples = []
        target_ntokens = 0

        for sample in samples:
            index = sample['id']
            indices.append(index)

            source_feature_samples.append(sample['source_features'])
            source_location_samples.append(sample['source_locations'])
            source_lengths.append(self.img_ds.sizes[index])

            target_samples.append(sample['target'])
            target_ntokens += self.cap_ds.sizes[index]

        num_sentences = len(samples)

        # FIXME: workaround for edge case in parallel processing
        # (framework passes empty samples list
        # to collater under certain conditions)
        if num_sentences == 0:
            return None

        indices = torch.tensor(indices, dtype=torch.long)

        source_feature_batch, source_location_batch = \
            self.img_ds.collater(list(zip(source_feature_samples, source_location_samples)))

        # TODO: switch depending on SCST or CE training
        if self.scst:
            target_batch = target_samples
            rotate_batch = None
        else:
            target_batch = data_utils.collate_tokens(target_samples,
                                                     pad_idx=self.cap_dict.pad(),
                                                     eos_idx=self.cap_dict.eos(),
                                                     move_eos_to_beginning=False)
            rotate_batch = data_utils.collate_tokens(target_samples,
                                                     pad_idx=self.cap_dict.pad(),
                                                     eos_idx=self.cap_dict.eos(),
                                                     move_eos_to_beginning=True)

        return {
            'id': indices,
            'net_input': {
                'src_tokens': source_feature_batch,
                'src_locations': source_location_batch,
                'src_lengths': source_lengths,
                'prev_output_tokens': rotate_batch,
            },
            'target': target_batch,
            'ntokens': target_ntokens,
            'nsentences': num_sentences,
        }
