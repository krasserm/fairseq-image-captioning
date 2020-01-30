import torch.nn as nn


class FeatureProjection(nn.Module):
    """
    Projects image features into a space of
    dimensionality `args.encoder_embed_dim`.
    """

    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(args.features_dim, args.encoder_embed_dim)

        # The following members are needed to
        # interface with TransformerEncoder.
        self.embedding_dim = args.encoder_embed_dim
        self.padding_idx = -1

    def forward(self, x):
        return self.linear(x)


class SpatialEncoding(nn.Module):
    """
    Encodes bounding box coordinates and relative sizes
    as vector of dimensionality `args.encoder_embed_dim`.
    """

    def __init__(self, args):
        super().__init__()
        self.linear = nn.Linear(5, args.encoder_embed_dim)

    def forward(self, x):
        return self.linear(x)
