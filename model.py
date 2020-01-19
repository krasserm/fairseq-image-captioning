import modules

import torch
import torch.nn.functional as F

from torchvision import models
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.inception import model_urls

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
        self.feature_projection = modules.FeatureProjection(args) \
            if not args.no_projection else None
        self.spatial_encoding = modules.SpatialEncoding(args) \
            if args.feature_spatial_encoding else None

    def forward(self, src_tokens, src_lengths, src_locations, **kwargs):
        x = src_tokens

        if self.feature_projection is not None:
            x = self.feature_projection(src_tokens)
        if self.spatial_encoding is not None:
            x += self.spatial_encoding(src_locations)

        # B x T x C -> T x B x C
        enc_out = x.transpose(0, 1)

        # compute padding mask
        enc_padding_mask = create_padding_mask(src_tokens, src_lengths)

        return transformer.EncoderOut(encoder_out=enc_out,
                                      encoder_padding_mask=enc_padding_mask,
                                      encoder_embedding=None,
                                      encoder_states=None)

    def reorder_encoder_out(self, encoder_out, new_order):
        enc_out = encoder_out.encoder_out
        enc_padding_mask = encoder_out.encoder_padding_mask

        return transformer.EncoderOut(encoder_out=enc_out.index_select(1, new_order),
                                      encoder_padding_mask=enc_padding_mask.index_select(0, new_order),
                                      encoder_embedding=None,
                                      encoder_states=None)


class TransformerCaptioningEncoder(transformer.TransformerEncoder):
    def __init__(self, args):
        super().__init__(args, None, modules.FeatureProjection(args))
        self.spatial_encoding = modules.SpatialEncoding(args) \
            if args.feature_spatial_encoding else None

    def forward(self, src_tokens, src_lengths, src_locations, **kwargs):
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)

        if self.spatial_encoding is not None:
            x += self.spatial_encoding(src_locations)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = create_padding_mask(src_tokens, src_lengths)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        return transformer.EncoderOut(encoder_out=x,
                                      encoder_padding_mask=encoder_padding_mask,
                                      encoder_embedding=None,
                                      encoder_states=None)


class CaptioningModel(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        transformer.TransformerModel.add_args(parser)
        parser.add_argument('--features-dim', type=int, default=2048,
                            help='visual features dimension')
        parser.add_argument('--feature-spatial-encoding', default=False, action='store_true',
                            help='use feature spatial encoding')

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

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)

        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

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
    args.encoder_layers = getattr(args, 'encoder_layers', 3)


@register_model_architecture('simplistic-captioning-model', 'simplistic-captioning-arch')
def simplistic_captioning_arch(args):
    if args.no_projection:
        args.encoder_embed_dim = args.features_dim


class Inception3Base(models.inception.Inception3):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        return model

    return Inception3Base(**kwargs)
