import numpy as np

import torch
import torch.nn.functional as F

from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.inception import model_urls, Inception3


def inception_v3_base(pretrained=False, progress=True, **kwargs):
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' in kwargs:
            original_aux_logits = kwargs['aux_logits']
            kwargs['aux_logits'] = True
        else:
            original_aux_logits = True
        model = Inception3Base(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['inception_v3_google'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.AuxLogits
    else:
        model = Inception3Base(**kwargs)

    del model.fc
    return model


class Inception3Base(Inception3):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grid_shape = (8, 8)
        self.grid_locations = self._grid_locations(self.grid_shape)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        return x

    @staticmethod
    def _grid_locations(grid_shape):
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
                    (j + 1) * rel_tile_w,
                    (i + 1) * rel_tile_h,
                    rel_tile_area
                ], dtype=np.float32)

        return torch.as_tensor(rel_tile_locations).view(num_tiles, 5)
