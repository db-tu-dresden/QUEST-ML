import argparse

from ml import Model, Config
from ml.models import get_model_from_type


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
        parser.add_argument('--decoder', type=str, help='Decoder model name')

    @classmethod
    def build_model(cls, config: Config):
        encoder_type = config['encoder'] if 'encoder' in config else None
        encoder_model = get_model_from_type(encoder_type, config)
        encoder = encoder_model.build_model(config)

        decoder_type = config['decoder'] if 'decoder' in config else None
        decoder_model = get_model_from_type(decoder_type, config)
        decoder = decoder_model.build_model(config)

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
        parser.add_argument('--fusion', type=str, help='Fusion model name')
        parser.add_argument('--decoder', type=str, help='Decoder model name')

    @classmethod
    def build_model(cls, config: Config):
        encoder_type = config['encoder'] if 'encoder' in config else None
        encoder_model = get_model_from_type(encoder_type, config)
        encoder = encoder_model.build_model(config)

        fusion_type = config['fusion'] if 'fusion' in config else None
        fusion_model = get_model_from_type(fusion_type, config)
        fusion = fusion_model.build_model(config)

        decoder_type = config['decoder'] if 'decoder' in config else None
        decoder_model = get_model_from_type(decoder_type, config)
        decoder = decoder_model.build_model(config)

        return cls(encoder, fusion, decoder)
