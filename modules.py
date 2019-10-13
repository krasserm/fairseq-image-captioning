import torch.nn as nn


class Projection(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(2048, args.encoder_embed_dim)
        self.embedding_dim = args.encoder_embed_dim
        self.padding_idx = -1

    def forward(self, x):
        return self.linear(x)
