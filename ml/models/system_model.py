import argparse

import torch
from torch import nn

from ml import Config, ddp
from ml.models import get_model_from_type, Model, register_model_architecture, register_model, ARCH_CONFIG_REGISTRY, \
    add_arch_args


class FusionModel(Model):
    def __init__(self, input_size: int, hidden_size: int, model: Model, dropout: float):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.model = model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        _, (out, _) = self.lstm(out)
        out = out.squeeze()
        out = self.model(out)
        out = self.dropout(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = ''):
        parser.add_argument(f'--{prefix}fusion_input_size', type=int, metavar='N', help='Input size')
        parser.add_argument(f'--{prefix}fusion_hidden_size', type=int, metavar='N', help='Hidden size')
        parser.add_argument(f'--{prefix}fusion_dropout', type=int, metavar='N', help='Dropout value')

        parser.add_argument('--fusion_model', type=str, help='Fusion model name. '
                                                             'To see fusion model specific arguments, use --help on '
                                                             'the model architecture. The default is mlp. Every '
                                                             'model parameter can be set by '
                                                             '--fusion_model_{PARAMETER}, '
                                                             'e.g. --fusion_model_hidden_size 32')

        root_parser = getattr(parser, 'root_parser', None)
        if root_parser:
            add_arch_args(root_parser, f'{prefix}fusion_model',
                          f'{prefix}fusion-model-specific configuration',
                          prefix=f'{prefix}fusion_model_')

    @classmethod
    def build_model(cls, config: Config, prefix: str = '') -> Model:
        model_type = config[f'{prefix}fusion_model'] if f'{prefix}fusion_model' in config else None
        model = get_model_from_type(model_type, config)
        model = model.build_model(config, f'{prefix}fusion_model_')

        return cls(config[f'{prefix}fusion_input_size'], config[f'{prefix}fusion_hidden_size'], model,
                   config[f'{prefix}fusion_dropout'])


class ProcessStateEncoder(Model):
    def __init__(self, model: Model):
        super().__init__()

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.model(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = 'encoder_'):
        root_parser = getattr(parser, 'root_parser', None)
        if root_parser:
            add_arch_args(root_parser, f'{prefix}model',
                          f'{prefix}model-specific configuration',
                          prefix=f'{prefix}model_')

    @classmethod
    def build_model(cls, config: Config, prefix: str = 'encoder_') -> Model:
        encoder_type = config[f'{prefix}model'] if f'{prefix}model' in config else None
        encoder_model = get_model_from_type(encoder_type, config)
        encoder = encoder_model.build_model(config, prefix)

        return cls(encoder)


class ProcessStateDecoder(Model):
    def __init__(self, model: Model):
        super().__init__()

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.model(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = 'decoder_'):
        root_parser = getattr(parser, 'root_parser', None)
        if root_parser:
            add_arch_args(root_parser, f'{prefix}model',
                          f'{prefix}model-specific configuration',
                          prefix=f'{prefix}process_decoder_')

    @classmethod
    def build_model(cls, config: Config, prefix: str = 'decoder_') -> Model:
        decoder_type = config[f'{prefix}model'] if f'{prefix}model' in config else None
        decoder_model = get_model_from_type(decoder_type, config)
        decoder = decoder_model.build_model(config, prefix)

        return cls(decoder)


class SystemStateEncoder(Model):
    def __init__(self, encoder: Model, fusion: Model, dropout: float):
        super().__init__()

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.fusion = fusion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.encoder(out)
        out = self.dropout(out)
        out = self.fusion(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = 'encoder_'):
        ProcessStateEncoder.add_args(parser, prefix)
        FusionModel.add_args(parser, prefix)

        parser.add_argument(f'--{prefix}dropout', type=int, metavar='N', help='Dropout value')

    @classmethod
    def build_model(cls, config: Config, prefix: str = 'encoder_') -> Model:
        encoder = ProcessStateEncoder.build_model(config, prefix)
        fusion = FusionModel.build_model(config, prefix)

        return cls(encoder, fusion, config[f'{prefix}dropout'])


class SystemStateDecoder(Model):
    def __init__(self, decoder: Model, processes: int):
        super().__init__()

        self.decoder = decoder
        self.processes = processes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        out = torch.stack([self.decoder(out + i / self.processes) for i in range(self.processes)])
        out = out.transpose(0, 1)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = 'decoder_'):
        ProcessStateDecoder.add_args(parser, prefix)

    @classmethod
    def build_model(cls, config: Config, prefix: str = 'decoder_') -> Model:
        decoder = ProcessStateDecoder.build_model(config, prefix)

        return cls(decoder, config['processes'])


class TransformationModel(Model):
    def __init__(self, transformation: Model):
        super().__init__()

        self.transformation = transformation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.transformation(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = 'transformation_'):
        root_parser = getattr(parser, 'root_parser', None)
        if root_parser:
            add_arch_args(root_parser, f'{prefix}model',
                          f'{prefix}model-specific configuration',
                          prefix=prefix)

    @classmethod
    def build_model(cls, config: Config, prefix: str = 'transformation_') -> Model:
        transformation_type = config[f'{prefix}model'] if f'{prefix}model' in config else None
        transformation_model = get_model_from_type(transformation_type, config)
        transformation = transformation_model.build_model(config, prefix)

        return cls(transformation)


@register_model('system_model')
class SystemModel(Model):
    def __init__(self, encoder: Model, transformation: Model, decoder: Model):
        super().__init__()
        self.encoder = encoder
        self.transformation = transformation
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.encoder(out)
        out = self.transformation(out)
        out = self.decoder(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = ''):
        parser.add_argument('--encoder', type=str, help='Encoder model name. '
                                                        'To see encoder specific arguments, use --help on the encoder '
                                                        'architecture. The default is mlp. Every encoder parameter can '
                                                        'be set by --encoder_{PARAMETER}, '
                                                        'e.g. --encoder_hidden_size 32')
        parser.add_argument('--transformation', type=str, help='Transformation model name. '
                                                        'To see transformation model specific arguments, use --help '
                                                        'on the fusion model architecture. The default is mlp. Every '
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
            add_arch_args(root_parser, 'transformation', 'Transformation model-specific configuration',
                          prefix='transformation_')
            add_arch_args(root_parser, 'decoder', 'Decoder model-specific configuration', prefix='decoder_')

        parser.add_argument('--load_encoder', default=False, action=argparse.BooleanOptionalAction,
                            help='Whether to load the encoder from the provided model state dict')
        parser.add_argument('--load_transformation', default=False, action=argparse.BooleanOptionalAction,
                            help='Whether to load the transformation model from the provided model state dict')
        parser.add_argument('--load_decoder', default=False, action=argparse.BooleanOptionalAction,
                            help='Whether to load the decoder from the provided model state dict')

    @classmethod
    def build_model(cls, config: Config, prefix: str = '') -> Model:
        encoder = SystemStateEncoder.build_model(config)
        transformation = TransformationModel.build_model(config)
        decoder = SystemStateDecoder.build_model(config)

        return cls(encoder, transformation, decoder)

    def save(self, config: Config):
        if not ddp.is_main_process() or not config['save_model']:
            return
        torch.save({
            'model': self.state_dict(),
            'encoder': self.encoder,
            'transformation': self.transformation,
            'decoder': self.decoder,
        }, config['model_save_path'])

    def load(self, config: Config):
        if not config['load_model'] and not config['load_encoder'] and not config['load_transformation'] \
                and not config['load_decoder']:
            return
        checkpoint = torch.load(config['model_load_path'])
        if config['load_model']:
            self.load_state_dict(checkpoint['model'])
            return
        if config['load_encoder']:
            self.encoder.load_state_dict(checkpoint['encoder'])
        if config['load_transformation']:
            self.transformation.load_state_dict(checkpoint['fusion'])
        if config['load_decoder']:
            self.decoder.load_state_dict(checkpoint['decoder'])


def process_encoder(cfg: Config, prefix: str = 'encoder_'):
    cfg[f'{prefix}hidden_layers'] = cfg[f'{prefix}hidden_layers'] if f'{prefix}hidden_layers' in cfg else 0
    cfg[f'{prefix}hidden_size'] = cfg[f'{prefix}hidden_size'] if f'{prefix}hidden_size' in cfg else cfg['embedding_size']
    cfg[f'{prefix}output_size'] = cfg[f'{prefix}output_size'] if f'{prefix}output_size' in cfg else cfg['embedding_size']

    cfg[f'{prefix}model'] = cfg[f'{prefix}model'] if f'{prefix}model' in cfg else 'mlp'
    ARCH_CONFIG_REGISTRY[cfg[f'{prefix}model']](cfg, prefix)


def fusion_model(cfg: Config, prefix: str = 'encoder_'):
    prefix += 'fusion_'

    cfg[f'{prefix}dropout'] = cfg[f'{prefix}dropout'] \
        if f'{prefix}dropout' in cfg else cfg['dropout']

    cfg[f'{prefix}input_size'] = cfg[f'{prefix}input_size'] if f'{prefix}input_size' in cfg else cfg['embedding_size']
    cfg[f'{prefix}hidden_size'] = cfg[f'{prefix}hidden_size'] if f'{prefix}hidden_size' in cfg else cfg['hidden_size']

    cfg[f'{prefix}model_input_size'] = cfg[f'{prefix}model_input_size'] \
        if f'{prefix}model_input_size' in cfg else cfg['hidden_size']
    cfg[f'{prefix}model_hidden_layers'] = cfg[f'{prefix}model_hidden_layers'] \
        if f'{prefix}model_hidden_layers' in cfg else 0
    cfg[f'{prefix}model_output_size'] = cfg[f'{prefix}model_output_size'] \
        if f'{prefix}model_output_size' in cfg else cfg['embedding_size']

    cfg[f'{prefix}model'] = cfg[f'{prefix}model'] if f'{prefix}model' in cfg else 'mlp'
    ARCH_CONFIG_REGISTRY[cfg[f'{prefix}model']](cfg, f'{prefix}model_')


def process_decoder(cfg: Config, prefix: str = 'decoder_'):
    cfg[f'{prefix}input_size'] = cfg[f'{prefix}input_size'] \
        if f'{prefix}input_size' in cfg else cfg['embedding_size']
    cfg[f'{prefix}hidden_layers'] = cfg[f'{prefix}hidden_layers'] \
        if f'{prefix}hidden_layers' in cfg else cfg['encoder_hidden_layers']
    cfg[f'{prefix}hidden_size'] = cfg[f'{prefix}hidden_size'] \
        if f'{prefix}hidden_size' in cfg else cfg['embedding_size']

    cfg[f'{prefix}model'] = cfg[f'{prefix}model'] if f'{prefix}model' in cfg else 'mlp'
    ARCH_CONFIG_REGISTRY[cfg[f'{prefix}model']](cfg, prefix)


def system_encoder(cfg: Config, prefix: str = 'encoder_'):
    cfg[f'{prefix}dropout'] = cfg[f'{prefix}dropout'] if f'{prefix}dropout' in cfg else 0.25

    process_encoder(cfg, prefix)
    fusion_model(cfg, prefix)


def system_transformation(cfg: Config, prefix: str = 'transformation_'):
    cfg[f'{prefix}input_size'] = cfg[f'{prefix}input_size'] \
        if f'{prefix}input_size' in cfg else cfg['embedding_size']
    cfg[f'{prefix}hidden_layers'] = cfg[f'{prefix}hidden_layers'] \
        if f'{prefix}hidden_layers' in cfg else cfg['hidden_layers']
    cfg[f'{prefix}output_size'] = cfg[f'{prefix}output_size'] \
        if f'{prefix}output_size' in cfg else cfg['embedding_size']

    cfg[f'{prefix}model'] = cfg[f'{prefix}encoder'] if f'{prefix}encoder' in cfg else 'mlp'
    ARCH_CONFIG_REGISTRY[cfg[f'{prefix}model']](cfg, prefix)


def system_decoder(cfg: Config, prefix: str = 'decoder_'):
    process_decoder(cfg, prefix)


@register_model_architecture('system_model', 'system_model')
def system_model(cfg: Config):
    cfg['dropout'] = cfg['dropout'] if 'dropout' in cfg else 0.25

    cfg['input_size'] = cfg['jobs'] if 'jobs' in cfg else 16
    cfg['embedding_size'] = cfg['embedding_size'] if 'embedding_size' in cfg else 16
    cfg['hidden_size'] = cfg['hidden_size'] if 'hidden_size' in cfg else 256
    cfg['hidden_layers'] = cfg['hidden_layers'] if 'hidden_layers' in cfg else 5
    cfg['output_size'] = cfg['output_size'] if 'output_size' in cfg else cfg['input_size']

    system_encoder(cfg, 'encoder_')
    system_transformation(cfg, 'transformation_')
    system_decoder(cfg, 'decoder_')