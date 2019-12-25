import torch.nn as nn


class FeatureEmbedding(nn.Module):
    """Models a learned embedding of visual features.
    """

    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(args.feature_dim, args.encoder_embed_dim)

        # The following members are needed to
        # interface with TransformerEncoder.
        self.embedding_dim = args.encoder_embed_dim
        self.padding_idx = -1

    def forward(self, x):
        return self.linear(x)


class SpatialEmbedding(nn.Module):
    """Models a learned embedding of spatial positions.
    """

    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(5, args.encoder_embed_dim)

    def forward(self, x):
        return self.linear(x)
