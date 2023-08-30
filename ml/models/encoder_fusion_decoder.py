import argparse

import torch

from ml import Config, ddp
from ml.models import get_model_from_type, Model, register_model_architecture, register_model, ARCH_CONFIG_REGISTRY, \
    add_arch_args


@register_model('encoder_fusion_decoder')
class EncoderFusionDecoder(Model):
    def __init__(self, encoder: Model, fusion: Model, decoder: Model):
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion
        self.decoder = decoder

    def forward(self, x):
        out = x
        out = self.encoder(out)
        out = self.fusion(out)
        out = self.decoder(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = ''):
        parser.add_argument('--encoder', type=str, help='Encoder model name. '
                                                        'To see encoder specific arguments, use --help on the encoder '
                                                        'architecture. The default is mlp. Every encoder parameter can '
                                                        'be set by --encoder_{PARAMETER}, '
                                                        'e.g. --encoder_hidden_size 32')
        parser.add_argument('--fusion', type=str, help='Fusion model name. '
                                                       'To see fusion model specific arguments, use --help on the '
                                                       'fusion model architecture. The default is mlp. Every '
                                                       'fusion model parameter can be set by --fusion_{PARAMETER}, '
                                                       'e.g. --fusion_hidden_size 32')
        parser.add_argument('--decoder', type=str, help='Decoder model name. '
                                                        'To see decoder specific arguments, use --help on the decoder '
                                                        'architecture. The default is mlp. Every decoder parameter can '
                                                        'be set by --decoder_{PARAMETER}, '
                                                        'e.g. --decoder_hidden_size 32')

        root_parser = getattr(parser, 'root_parser', None)
        if root_parser:
            add_arch_args(root_parser, 'encoder', 'Encoder model-specific configuration', prefix='encoder_')
            add_arch_args(root_parser, 'fusion', 'Fusion model-specific configuration', prefix='fusion_')
            add_arch_args(root_parser, 'decoder', 'Decoder model-specific configuration', prefix='decoder_')

        parser.add_argument('--load_encoder', default=False, action=argparse.BooleanOptionalAction,
                            help='Whether to load the encoder from the provided model state dict')
        parser.add_argument('--load_fusion', default=False, action=argparse.BooleanOptionalAction,
                            help='Whether to load the fusion model from the provided model state dict')
        parser.add_argument('--load_decoder', default=False, action=argparse.BooleanOptionalAction,
                            help='Whether to load the decoder from the provided model state dict')

    @classmethod
    def build_model(cls, config: Config, prefix: str = ''):
        encoder_type = config['encoder'] if 'encoder' in config else None
        encoder_model = get_model_from_type(encoder_type, config)
        encoder = encoder_model.build_model(config, 'encoder_')

        fusion_type = config['fusion'] if 'fusion' in config else None
        fusion_model = get_model_from_type(fusion_type, config)
        fusion = fusion_model.build_model(config, 'fusion_')

        decoder_type = config['decoder'] if 'decoder' in config else None
        decoder_model = get_model_from_type(decoder_type, config)
        decoder = decoder_model.build_model(config, 'decoder_')

        return cls(encoder, fusion, decoder)

    def save(self, config: Config):
        if not ddp.is_main_process() or not config['save_model']:
            return
        torch.save({
            'model': self.state_dict(),
            'encoder': self.encoder.state_dict(),
            'fusion': self.fusion.state_dict(),
            'decoder': self.decoder.state_dict(),
        }, self.get_save_path(config))

    def load(self, config: Config):
        if not config['load_model'] and not config['load_encoder'] and not config['load_fusion'] \
                and not config['load_decoder']:
            return
        checkpoint = torch.load(self.get_load_path(config))
        if config['load_model']:
            self.load_state_dict(checkpoint['model'])
            return
        if config['load_encoder']:
            self.encoder.load_state_dict(checkpoint['encoder'])
        if config['load_fusion']:
            self.fusion.load_state_dict(checkpoint['fusion'])
        if config['load_decoder']:
            self.decoder.load_state_dict(checkpoint['decoder'])


@register_model_architecture('encoder_fusion_decoder', 'encoder_fusion_decoder_mlp')
def encoder_fusion_decoder(cfg: Config):
    cfg['encoder'] = cfg['encoder'] if 'encoder' in cfg else 'mlp'
    cfg['fusion'] = cfg['fusion'] if 'fusion' in cfg else 'mlp'
    cfg['decoder'] = cfg['decoder'] if 'decoder' in cfg else 'mlp'

    ARCH_CONFIG_REGISTRY[cfg['encoder']](cfg, 'encoder_')
    ARCH_CONFIG_REGISTRY[cfg['fusion']](cfg, 'fusion_')
    ARCH_CONFIG_REGISTRY[cfg['decoder']](cfg, 'decoder_')
