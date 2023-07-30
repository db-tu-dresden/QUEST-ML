import argparse

import torch

from ml import Config, ddp
from ml.models import get_model_from_type, Model, register_model_architecture, register_model


@register_model('encoder_decoder')
class EncoderDecoder(Model):
    def __init__(self, encoder: Model, decoder: Model):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        out = x
        out = self.encoder(out)
        out = self.decoder(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--encoder', type=str, help='Encoder model name')
        parser.add_argument('--encoder_state_path', type=str,
                            help='Path to the state dict of a pretrained encoder model')
        parser.add_argument('--decoder', type=str, help='Decoder model name')
        parser.add_argument('--decoder_state_path', type=str,
                            help='Path to the state dict of a pretrained decoder model')

        parser.add_argument('--load_encoder', default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--load_decoder', default=False, action=argparse.BooleanOptionalAction)

    @classmethod
    def build_model(cls, config: Config):
        encoder_type = config['encoder'] if 'encoder' in config else None
        encoder_model = get_model_from_type(encoder_type, config)
        encoder = encoder_model.build_model(config)
        if config['encoder_state_path']:
            checkpoint = torch.load(config['encoder_state_path'])
            encoder.load_state_dict(checkpoint['model'])

        decoder_type = config['decoder'] if 'decoder' in config else None
        decoder_model = get_model_from_type(decoder_type, config)
        decoder = decoder_model.build_model(config)
        if config['decoder_state_path']:
            checkpoint = torch.load(config['decoder_state_path'])
            encoder.load_state_dict(checkpoint['model'])

        return cls(encoder, decoder)

    def save(self, config: Config):
        if ddp.is_main_process():
            if config['save_model']:
                torch.save({
                    'model': self.state_dict(),
                    'encoder': self.encoder,
                    'decoder': self.decoder,
                }, config['model_save_path'])

    def load(self, config: Config):
        checkpoint = torch.load(config['model_load_path'])
        if config['load_model']:
            self.load_state_dict(checkpoint['model'])
            return
        if config['load_encoder']:
            self.encoder.load_state_dict(checkpoint['encoder'])
        if config['load_decoder']:
            self.decoder.load_state_dict(checkpoint['decoder'])


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
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--encoder', type=str, help='Encoder model name')
        parser.add_argument('--encoder_state_path', type=str,
                            help='Path to the state dict of a pretrained encoder model')
        parser.add_argument('--fusion', type=str, help='Fusion model name')
        parser.add_argument('--fusion_state_path', type=str,
                            help='Path to the state dict of a pretrained fusion model')
        parser.add_argument('--decoder', type=str, help='Decoder model name')
        parser.add_argument('--decoder_state_path', type=str,
                            help='Path to the state dict of a pretrained decoder model')

        parser.add_argument('--load_encoder', default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--load_fusion', default=False, action=argparse.BooleanOptionalAction)
        parser.add_argument('--load_decoder', default=False, action=argparse.BooleanOptionalAction)

    @classmethod
    def build_model(cls, config: Config):
        encoder_type = config['encoder'] if 'encoder' in config else None
        encoder_model = get_model_from_type(encoder_type, config)
        encoder = encoder_model.build_model(config)
        if config['encoder_state_path']:
            checkpoint = torch.load(config['encoder_state_path'])
            encoder.load_state_dict(checkpoint['model'])

        fusion_type = config['fusion'] if 'fusion' in config else None
        fusion_model = get_model_from_type(fusion_type, config)
        fusion = fusion_model.build_model(config)
        if config['fusion_state_path']:
            checkpoint = torch.load(config['fusion_state_path'])
            encoder.load_state_dict(checkpoint['model'])

        decoder_type = config['decoder'] if 'decoder' in config else None
        decoder_model = get_model_from_type(decoder_type, config)
        decoder = decoder_model.build_model(config)
        if config['decoder_state_path']:
            checkpoint = torch.load(config['decoder_state_path'])
            encoder.load_state_dict(checkpoint['model'])

        return cls(encoder, fusion, decoder)

    def save(self, config: Config):
        if ddp.is_main_process():
            if config['save_model']:
                torch.save({
                    'model': self.state_dict(),
                    'encoder': self.encoder,
                    'fusion': self.fusion,
                    'decoder': self.decoder,
                }, config['model_save_path'])

    def load(self, config: Config):
        checkpoint = torch.load(config['model_load_path'])
        if config['load_model']:
            self.load_state_dict(checkpoint['model'])
            return
        if config['load_encoder']:
            self.encoder.load_state_dict(checkpoint['encoder'])
        if config['load_fusion']:
            self.encoder.load_state_dict(checkpoint['fusion'])
        if config['load_decoder']:
            self.decoder.load_state_dict(checkpoint['decoder'])


@register_model_architecture('encoder_decoder', 'encoder_decoder_mlp')
def encoder_decoder(cfg: Config):
    cfg['encoder'] = cfg['encoder'] if 'encoder' in cfg else 'mlp'
    cfg['decoder'] = cfg['decoder'] if 'decoder' in cfg else 'mlp'


@register_model_architecture('encoder_fusion_decoder', 'encoder_fusion_decoder_mlp')
def encoder_fusion_decoder(cfg: Config):
    cfg['encoder'] = cfg['encoder'] if 'encoder' in cfg else 'mlp'
    cfg['fusion'] = cfg['fusion'] if 'fusion' in cfg else 'mlp'
    cfg['decoder'] = cfg['decoder'] if 'decoder' in cfg else 'mlp'
