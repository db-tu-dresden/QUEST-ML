import argparse

import torch

from ml import Config
from ml.models import get_model_from_type, Model


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

    @classmethod
    def build_model(cls, config: Config):
        encoder_type = config['encoder'] if 'encoder' in config else None
        encoder_model = get_model_from_type(encoder_type, config)
        encoder = encoder_model.build_model(config)
        if config['encoder_state_path']:
            encoder.load_state_dict(torch.load(config['encoder_state_path']))

        decoder_type = config['decoder'] if 'decoder' in config else None
        decoder_model = get_model_from_type(decoder_type, config)
        decoder = decoder_model.build_model(config)
        if config['decoder_state_path']:
            encoder.load_state_dict(torch.load(config['decoder_state_path']))

        return cls(encoder, decoder)


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

    @classmethod
    def build_model(cls, config: Config):
        encoder_type = config['encoder'] if 'encoder' in config else None
        encoder_model = get_model_from_type(encoder_type, config)
        encoder = encoder_model.build_model(config)
        if config['encoder_state_path']:
            encoder.load_state_dict(torch.load(config['encoder_state_path']))

        fusion_type = config['fusion'] if 'fusion' in config else None
        fusion_model = get_model_from_type(fusion_type, config)
        fusion = fusion_model.build_model(config)
        if config['fusion_state_path']:
            encoder.load_state_dict(torch.load(config['fusion_state_path']))

        decoder_type = config['decoder'] if 'decoder' in config else None
        decoder_model = get_model_from_type(decoder_type, config)
        decoder = decoder_model.build_model(config)
        if config['decoder_state_path']:
            encoder.load_state_dict(torch.load(config['decoder_state_path']))

        return cls(encoder, fusion, decoder)
