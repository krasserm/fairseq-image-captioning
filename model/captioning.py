import torch
import torch.nn.functional as F

from modules import FeatureEmbedding, SpatialEmbedding
from model.inception import inception_v3_base

from fairseq.models import FairseqEncoder, BaseFairseqModel
from fairseq.models import register_model, register_model_architecture, transformer


def create_padding_mask(src_tokens, src_lengths):
    padding_mask = torch.zeros(src_tokens.shape[:2],
                               dtype=torch.bool,
                               device=src_tokens.device)

    for i, src_length in enumerate(src_lengths):
        padding_mask[i, src_length:] = 1

    return padding_mask


class SimplisticCaptioningEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(dictionary=None)
        self.feature_dim = args.feature_dim
        # TODO: make pretrained configurable
        self.inception = inception_v3_base(pretrained=True, aux_logits=False)
        self.locations = torch.nn.Parameter(self.inception.grid_locations, requires_grad=False)

        self.feature_embedding = FeatureEmbedding(args) \
            if not args.no_projection else None
        self.location_embedding = SpatialEmbedding(args) \
            if args.feature_spatial_embeddings else None

        for param in self.inception.parameters():
            # TODO: make configurable
            param.requires_grad = True

    def forward(self, source, **kwargs):
        x = self.inception(source).permute(0, 2, 3, 1).view(-1, self.locations.shape[0], self.feature_dim)

        if self.feature_embedding is not None:
            x = self.feature_embedding(x)
        if self.location_embedding is not None:
            x += self.location_embedding(self.locations)

        # compute padding mask
        encoder_padding_mask = create_padding_mask(x, [self.locations.shape[0]] * source.shape[0])

        # B x T x C -> T x B x C
        encoder_out = x.transpose(0, 1)

        return {
            'encoder_out': encoder_out,
            'encoder_padding_mask': encoder_padding_mask
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        enc_out = encoder_out['encoder_out']
        enc_padding_mask = encoder_out['encoder_padding_mask']

        return {
            'encoder_out': enc_out.index_select(1, new_order),
            'encoder_padding_mask': enc_padding_mask.index_select(0, new_order)
        }


class TransformerCaptioningEncoder(transformer.TransformerEncoder):
    def __init__(self, args):
        super().__init__(args, None, FeatureEmbedding(args))
        self.feature_dim = args.feature_dim
        # TODO: make pretrained configurable
        self.inception = inception_v3_base(pretrained=True, aux_logits=False)
        self.locations = torch.nn.Parameter(self.inception.grid_locations, requires_grad=False)

        self.location_embedding = SpatialEmbedding(args) \
            if args.feature_spatial_embeddings else None

        for param in self.inception.parameters():
            # TODO: make configurable
            param.requires_grad = True

    def forward(self, source, **kwargs):
        x = self.inception(source).permute(0, 2, 3, 1).view(-1, self.locations.shape[0], self.feature_dim)

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(x)

        if self.location_embedding is not None:
            x += self.location_embedding(self.locations)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # compute padding mask
        encoder_padding_mask = create_padding_mask(x, [self.locations.shape[0]] * source.shape[0])

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }


class CaptioningModel(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        transformer.TransformerModel.add_args(parser)
        parser.add_argument('--feature-dim', type=int, default=2048,
                            help='visual features dimension')
        parser.add_argument('--feature-spatial-embeddings', default=False, action='store_true',
                            help='use feature spatial embeddings')

    @classmethod
    def build_model(cls, args, task):
        transformer.base_architecture(args)

        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = transformer.DEFAULT_MAX_TARGET_POSITIONS

        captions_dict = task.target_dictionary

        encoder = cls.do_build_encoder(args)
        decoder = cls.do_build_decoder(args, captions_dict)
        return cls.do_build_model(encoder, decoder)

    @classmethod
    def do_build_model(cls, encoder, decoder):
        raise NotImplementedError

    @classmethod
    def do_build_encoder(cls, args):
        raise NotImplementedError

    @classmethod
    def do_build_decoder(cls, args, captions_dict):
        decoder_embedding = transformer.Embedding(num_embeddings=len(captions_dict),
                                                  embedding_dim=args.decoder_embed_dim,
                                                  padding_idx=captions_dict.pad())
        return transformer.TransformerDecoder(args, captions_dict, decoder_embedding)

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(source, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)

        return decoder_out

    def max_decoder_positions(self):
        return self.decoder.max_positions()


@register_model('default-captioning-model')
class DefaultCaptioningModel(CaptioningModel):
    @classmethod
    def do_build_encoder(cls, args):
        return TransformerCaptioningEncoder(args)

    @classmethod
    def do_build_model(cls, encoder, decoder):
        return DefaultCaptioningModel(encoder, decoder)


@register_model('simplistic-captioning-model')
class SimplisticCaptioningModel(CaptioningModel):
    @staticmethod
    def add_args(parser):
        CaptioningModel.add_args(parser)
        parser.add_argument('--no-projection', default=False, action='store_true',
                            help='do not project visual features')

    @classmethod
    def do_build_encoder(cls, args):
        return SimplisticCaptioningEncoder(args)

    @classmethod
    def do_build_model(cls, encoder, decoder):
        return SimplisticCaptioningModel(encoder, decoder)


@register_model_architecture('default-captioning-model', 'default-captioning-arch')
def default_captioning_arch(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 2)


@register_model_architecture('simplistic-captioning-model', 'simplistic-captioning-arch')
def simplistic_captioning_arch(args):
    if args.no_projection:
        args.encoder_embed_dim = args.features_dim
