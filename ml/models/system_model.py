import argparse

import torch
from torch import nn

from ml import Config, ddp
from ml.models import get_model_from_type, Model, register_model_architecture, register_model, ARCH_CONFIG_REGISTRY, \
    add_arch_args


def unsqueeze1(x: torch.Tensor):
    return torch.unsqueeze(x, dim=1)


class FusionModel(Model):
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool, model: Model, dropout: float,
                 maxpool_states: bool):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.model = model
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.maxpool_states = maxpool_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        shape = out.shape

        out, (h_0, c_0) = self.lstm(out)

        if self.maxpool_states:
            out = nn.functional.max_pool2d(out, (shape[1], 1)).squeeze(1)
        else:
            out = h_0.transpose(0, 1).reshape((shape[0], -1))

        out = self.model(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = 'encoder_fusion_'):
        parser.add_argument(f'--{prefix}input_size', type=int, metavar='N', help='Input size')
        parser.add_argument(f'--{prefix}hidden_size', type=int, metavar='N', help='Hidden size')
        parser.add_argument(f'--{prefix}dropout', type=int, metavar='N', help='Dropout value')

        parser.add_argument(f'--{prefix}maxpool_last_layer_hidden_states',
                            action=argparse.BooleanOptionalAction,
                            help='Dropout value')

        parser.add_argument('--fusion_model', type=str, help='Fusion model name. '
                                                             'To see fusion model specific arguments, use --help on '
                                                             'the model architecture. The default is mlp. Every '
                                                             'model parameter can be set by '
                                                             '--fusion_model_{PARAMETER}, '
                                                             'e.g. --fusion_model_hidden_size 32')

        root_parser = getattr(parser, 'root_parser', None)
        if root_parser:
            add_arch_args(root_parser, f'{prefix}model',
                          f'{prefix}model-specific configuration',
                          prefix=f'{prefix}model_', default='mlp')

    @classmethod
    def build_model(cls, config: Config, prefix: str = 'encoder_fusion_') -> Model:
        model_type = config[f'{prefix}model'] if f'{prefix}model' in config else None
        model = get_model_from_type(model_type, config)
        model = model.build_model(config, f'{prefix}model_')

        return cls(config[f'{prefix}input_size'], config[f'{prefix}hidden_size'],
                   config[f'{prefix}bidirectional'], model, config[f'{prefix}dropout'],
                   config[f'{prefix}maxpool_last_layer_hidden_states'])


class DeFusionModel(Model):
    def __init__(self, input_size: int, hidden_size: int, model: Model, dropout: float, processes):
        super().__init__()
        self.model = model
        self.activation = nn.ReLU()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.processes = processes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        device = out.device

        out = self.model(out)
        out = self.activation(out)
        out = self.dropout(out)

        shape = out.shape

        _input = torch.zeros((shape[0], 1, shape[1])).to(device)
        h_0 = out.unsqueeze(dim=0)
        c_0 = torch.zeros((1, shape[0], self.lstm.hidden_size)).to(device)

        _outs = []

        for i in range(self.processes):
            _out, (h_0, c_0) = self.lstm(_input, (h_0, c_0))
            _input = _out.detach()
            _outs.append(_out.squeeze())

        out = torch.stack(_outs).transpose(0, 1)
        out = self.dropout(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = 'decoder_defusion_'):
        parser.add_argument(f'--{prefix}input_size', type=int, metavar='N', help='Input size')
        parser.add_argument(f'--{prefix}hidden_size', type=int, metavar='N', help='Hidden size')
        parser.add_argument(f'--{prefix}dropout', type=int, metavar='N', help='Dropout value')

        parser.add_argument('--defusion_model', type=str, help='Defusion model name. '
                                                               'To see defusion model specific arguments, use --help '
                                                               'on the model architecture. The default is mlp. Every '
                                                               'model parameter can be set by '
                                                               '--defusion_model_{PARAMETER}, '
                                                               'e.g. --defusion_model_hidden_size 32')

        root_parser = getattr(parser, 'root_parser', None)
        if root_parser:
            add_arch_args(root_parser, f'{prefix}model',
                          f'{prefix}model-specific configuration',
                          prefix=f'{prefix}model_', default='mlp')

    @classmethod
    def build_model(cls, config: Config, prefix: str = 'decoder_defusion_') -> Model:
        model_type = config[f'{prefix}model'] if f'{prefix}model' in config else None
        model = get_model_from_type(model_type, config)
        model = model.build_model(config, f'{prefix}model_')

        return cls(config[f'{prefix}input_size'], config[f'{prefix}hidden_size'], model,
                   config[f'{prefix}dropout'], config[f'processes'])


class ProcessStateEncoder(Model):
    def __init__(self, model: Model):
        super().__init__()

        self.model = model
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.model(out)
        out = self.activation(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = 'process_encoder_'):
        root_parser = getattr(parser, 'root_parser', None)
        if root_parser:
            add_arch_args(root_parser, f'{prefix}model',
                          f'{prefix}model-specific configuration',
                          prefix=f'{prefix}model_', default='mlp')

    @classmethod
    def build_model(cls, config: Config, prefix: str = 'process_encoder_') -> Model:
        encoder_type = config[f'{prefix}model'] if f'{prefix}model' in config else None
        encoder_model = get_model_from_type(encoder_type, config)
        encoder = encoder_model.build_model(config, prefix)

        return cls(encoder)


class ProcessStateDecoder(Model):
    def __init__(self, model: Model):
        super().__init__()

        self.model = model
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.model(out)
        out = self.activation(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = 'process_decoder_'):
        root_parser = getattr(parser, 'root_parser', None)
        if root_parser:
            add_arch_args(root_parser, f'{prefix}model',
                          f'{prefix}model-specific configuration',
                          prefix=f'{prefix}model_', default='mlp')

    @classmethod
    def build_model(cls, config: Config, prefix: str = 'process_decoder_') -> Model:
        decoder_type = config[f'{prefix}model'] if f'{prefix}model' in config else None
        decoder_model = get_model_from_type(decoder_type, config)
        decoder = decoder_model.build_model(config, prefix)

        return cls(decoder)


class SystemStateEncoder(Model):
    def __init__(self, encoder: Model, fusion: Model, dropout: float, only_process: bool):
        super().__init__()

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout) if not only_process else nn.Identity()
        self.fusion = fusion if not only_process else torch.squeeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.encoder(out)
        out = self.dropout(out)
        out = self.fusion(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = 'encoder_'):
        ProcessStateEncoder.add_args(parser)
        FusionModel.add_args(parser)

        parser.add_argument(f'--{prefix}dropout', type=int, metavar='N', help='Dropout value')

    @classmethod
    def build_model(cls, config: Config, prefix: str = 'encoder_') -> Model:
        encoder = ProcessStateEncoder.build_model(config)
        fusion = FusionModel.build_model(config)

        return cls(encoder, fusion, config[f'{prefix}dropout'], config['only_process'])


class SystemStateDecoder(Model):
    def __init__(self, defusion: Model, decoder: Model, only_process: bool):
        super().__init__()

        self.defusion = defusion if not only_process else unsqueeze1
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.defusion(out)
        out = self.decoder(out)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, prefix: str = 'decoder_'):
        ProcessStateDecoder.add_args(parser)

    @classmethod
    def build_model(cls, config: Config, prefix: str = 'decoder_') -> Model:
        defusion = DeFusionModel.build_model(config)
        decoder = ProcessStateDecoder.build_model(config)

        return cls(defusion, decoder, config['only_process'])


class TransformationModel(Model):
    def __init__(self, transformation: Model):
        super().__init__()

        self.transformation = transformation
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        out = self.transformation(out)
        out = self.activation(out)
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
    def __init__(self, encoder: Model, transformation: Model, decoder: Model, config: Config):
        super().__init__()
        self.encoder = encoder
        self.transformation = transformation if not config['only_process'] and not config['only_system'] \
            else nn.Identity()
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
            SystemStateEncoder.add_args(parser)
            TransformationModel.add_args(parser)
            SystemStateDecoder.add_args(parser)

        parser.add_argument('--load_process_encoder', default=False, action=argparse.BooleanOptionalAction,
                            help='Whether to load the process state encoder from the provided model state dict')
        parser.add_argument('--load_encoder', default=False, action=argparse.BooleanOptionalAction,
                            help='Whether to load the encoder from the provided model state dict')
        parser.add_argument('--load_transformation', default=False, action=argparse.BooleanOptionalAction,
                            help='Whether to load the transformation model from the provided model state dict')
        parser.add_argument('--load_process_decoder', default=False, action=argparse.BooleanOptionalAction,
                            help='Whether to load the process state decoder from the provided model state dict')
        parser.add_argument('--load_decoder', default=False, action=argparse.BooleanOptionalAction,
                            help='Whether to load the decoder from the provided model state dict')

        parser.add_argument('--freeze', default=None, action=argparse.BooleanOptionalAction,
                            help='Whether to freeze the model')
        parser.add_argument('--freeze_process_encoder', default=None, action=argparse.BooleanOptionalAction,
                            help='Whether to freeze the process state encoder')
        parser.add_argument('--freeze_encoder', default=None, action=argparse.BooleanOptionalAction,
                            help='Whether to freeze the encoder')
        parser.add_argument('--freeze_transformation', default=None, action=argparse.BooleanOptionalAction,
                            help='Whether to freeze the transformation model')
        parser.add_argument('--freeze_process_decoder', default=None, action=argparse.BooleanOptionalAction,
                            help='Whether to freeze the process state decoder')
        parser.add_argument('--freeze_decoder', default=None, action=argparse.BooleanOptionalAction,
                            help='Whether to freeze the decoder')

        parser.add_argument('--only_process', default=False, action=argparse.BooleanOptionalAction,
                            help='Whether to only encode and decode process states')
        parser.add_argument('--only_system', default=False, action=argparse.BooleanOptionalAction,
                            help='Whether to only encode and decode system states')

        parser.add_argument('--process_autoencoder', default=False, action=argparse.BooleanOptionalAction,
                            help='Preset for --only_process --freeze --freeze_encoder --freeze_decoder '
                                 '--no-freeze_process_encoder --no-freeze_process_decoder --offset 0')
        parser.add_argument('--system_autoencoder', default=False, action=argparse.BooleanOptionalAction,
                            help='Preset for --only_system --freeze --no-freeze_encoder --no-freeze_decoder '
                                 '--freeze_process_encoder --no-freeze_process_decoder --offset 0')

    @classmethod
    def build_model(cls, config: Config, prefix: str = '') -> Model:
        assert not (config['only_process'] and config['only_system'])

        encoder = SystemStateEncoder.build_model(config)
        transformation = TransformationModel.build_model(config)
        decoder = SystemStateDecoder.build_model(config)

        model = cls(encoder, transformation, decoder, config)

        model.requires_grad_(not config['freeze'])
        model.encoder.requires_grad_(not config['freeze_encoder'])
        model.decoder.requires_grad_(not config['freeze_decoder'])
        model.transformation.requires_grad_(not config['freeze_transformation'])
        model.encoder.encoder.requires_grad_(not config['freeze_process_encoder'])
        model.decoder.decoder.requires_grad_(not config['freeze_process_decoder'])

        return model

    def save(self, config: Config):
        if not ddp.is_main_process() or not config['save_model']:
            return
        torch.save({
            'model': self.state_dict(),
            'process_state_encoder': self.encoder.encoder.state_dict(),
            'system_state_encoder': self.encoder.state_dict(),
            'transformation': self.transformation.state_dict(),
            'process_state_decoder': self.decoder.decoder.state_dict(),
            'system_state_decoder': self.decoder.state_dict(),
        }, self.get_save_path(config))

    def load(self, config: Config):
        if (not config['load_model'] and not config['load_transformation'] and
                not config['load_process_encoder'] and not config['load_encoder'] and
                not config['load_process_decoder'] and not config['load_decoder']):
            return
        checkpoint = torch.load(self.get_load_path(config))
        if config['load_model']:
            self.load_state_dict(checkpoint['model'])
            return
        if config['load_process_encoder']:
            self.encoder.encoder.load_state_dict(checkpoint['process_state_encoder'])
        if config['load_encoder']:
            self.encoder.load_state_dict(checkpoint['system_state_encoder'])
        if config['load_transformation']:
            self.transformation.load_state_dict(checkpoint['fusion'])
        if config['load_process_decoder']:
            self.decoder.decoder.load_state_dict(checkpoint['process_state_decoder'])
        if config['load_decoder']:
            self.decoder.load_state_dict(checkpoint['system_state_decoder'])


def process_encoder(cfg: Config, prefix: str = 'process_encoder_'):
    cfg[f'{prefix}hidden_layers'] = cfg[f'{prefix}hidden_layers'] if f'{prefix}hidden_layers' in cfg else 0
    cfg[f'{prefix}hidden_size'] = cfg[f'{prefix}hidden_size'] if f'{prefix}hidden_size' in cfg \
        else cfg['process_embedding_size']
    cfg[f'{prefix}output_size'] = cfg[f'{prefix}output_size'] if f'{prefix}output_size' in cfg \
        else cfg['process_embedding_size']
    cfg[f'{prefix}dropout'] = cfg[f'{prefix}dropout'] if f'{prefix}dropout' in cfg else 0.0

    cfg[f'{prefix}model'] = cfg[f'{prefix}model'] if f'{prefix}model' in cfg else 'mlp'
    ARCH_CONFIG_REGISTRY[cfg[f'{prefix}model']](cfg, prefix)

    cfg['freeze_process_encoder'] = (cfg['freeze_process_encoder'] if cfg['freeze_process_encoder'] is not None
                                     else cfg['freeze_encoder']) \
        if 'freeze_process_decoder' in cfg else False


def fusion_model(cfg: Config, prefix: str = 'encoder_fusion_'):
    cfg[f'{prefix}dropout'] = cfg[f'{prefix}dropout'] \
        if f'{prefix}dropout' in cfg else cfg['dropout']

    cfg[f'{prefix}input_size'] = cfg[f'{prefix}input_size'] if f'{prefix}input_size' in cfg \
        else cfg['process_embedding_size']
    cfg[f'{prefix}hidden_size'] = cfg[f'{prefix}hidden_size'] if f'{prefix}hidden_size' in cfg \
        else cfg['process_embedding_size']
    cfg[f'{prefix}bidirectional'] = cfg[f'{prefix}bidirectional'] if f'{prefix}bidirectional' in cfg else True

    cfg[f'{prefix}maxpool_last_layer_hidden_states'] = cfg[f'{prefix}maxpool_last_layer_hidden_states'] \
        if f'{prefix}maxpool_last_layer_hidden_states' in cfg else False

    cfg[f'{prefix}model_input_size'] = cfg[f'{prefix}model_input_size'] \
        if f'{prefix}model_input_size' in cfg else 2**int(cfg[f'{prefix}bidirectional']) * cfg[f'{prefix}hidden_size']
    cfg[f'{prefix}model_hidden_size'] = cfg[f'{prefix}model_hidden_size'] \
        if f'{prefix}model_hidden_size' in cfg else cfg['system_embedding_size']
    cfg[f'{prefix}model_hidden_layers'] = cfg[f'{prefix}model_hidden_layers'] \
        if f'{prefix}model_hidden_layers' in cfg else 0
    cfg[f'{prefix}model_output_size'] = cfg[f'{prefix}model_output_size'] \
        if f'{prefix}model_output_size' in cfg else cfg['system_embedding_size']

    cfg[f'{prefix}model'] = cfg[f'{prefix}model'] if f'{prefix}model' in cfg else 'mlp'
    ARCH_CONFIG_REGISTRY[cfg[f'{prefix}model']](cfg, f'{prefix}model_')


def defusion_model(cfg: Config, prefix: str = 'decoder_defusion_'):
    cfg[f'{prefix}dropout'] = cfg[f'{prefix}dropout'] if f'{prefix}dropout' in cfg else cfg['dropout']

    cfg[f'{prefix}input_size'] = cfg[f'{prefix}input_size'] if f'{prefix}input_size' in cfg \
        else cfg['process_embedding_size']
    cfg[f'{prefix}hidden_size'] = cfg[f'{prefix}hidden_size'] if f'{prefix}hidden_size' in cfg \
        else cfg['process_embedding_size']

    cfg[f'{prefix}model_input_size'] = cfg[f'{prefix}model_input_size'] \
        if f'{prefix}model_input_size' in cfg else cfg['system_embedding_size']
    cfg[f'{prefix}model_hidden_size'] = cfg[f'{prefix}model_hidden_size'] \
        if f'{prefix}model_hidden_size' in cfg else cfg[f'encoder_fusion_model_hidden_size']
    cfg[f'{prefix}model_hidden_layers'] = cfg[f'{prefix}model_hidden_layers'] \
        if f'{prefix}model_hidden_layers' in cfg else cfg['encoder_fusion_model_hidden_layers']
    cfg[f'{prefix}model_output_size'] = cfg[f'{prefix}model_output_size'] \
        if f'{prefix}model_output_size' in cfg else cfg['process_embedding_size']

    cfg[f'{prefix}model'] = cfg[f'{prefix}model'] if f'{prefix}model' in cfg else 'mlp'
    ARCH_CONFIG_REGISTRY[cfg[f'{prefix}model']](cfg, f'{prefix}model_')


def process_decoder(cfg: Config, prefix: str = 'process_decoder_'):
    cfg[f'{prefix}input_size'] = cfg[f'{prefix}input_size'] \
        if f'{prefix}input_size' in cfg else cfg['process_embedding_size']
    cfg[f'{prefix}hidden_layers'] = cfg[f'{prefix}hidden_layers'] \
        if f'{prefix}hidden_layers' in cfg else cfg['process_encoder_hidden_layers']
    cfg[f'{prefix}hidden_size'] = cfg[f'{prefix}hidden_size'] \
        if f'{prefix}hidden_size' in cfg else cfg['process_embedding_size']
    cfg[f'{prefix}dropout'] = cfg[f'{prefix}dropout'] if f'{prefix}dropout' in cfg else 0.0

    cfg[f'{prefix}model'] = cfg[f'{prefix}model'] if f'{prefix}model' in cfg else 'mlp'
    ARCH_CONFIG_REGISTRY[cfg[f'{prefix}model']](cfg, prefix)

    cfg['freeze_process_decoder'] = (cfg['freeze_process_decoder'] if cfg['freeze_process_decoder'] is not None
                                     else cfg['freeze_decoder']) \
        if 'freeze_process_decoder' in cfg else False


def system_encoder(cfg: Config, prefix: str = 'encoder_'):
    cfg[f'{prefix}dropout'] = cfg[f'{prefix}dropout'] if f'{prefix}dropout' in cfg else cfg['dropout']

    cfg['freeze_encoder'] = (cfg['freeze_encoder'] if cfg['freeze_encoder'] is not None else cfg['freeze']) \
        if 'freeze_encoder' in cfg else False

    process_encoder(cfg)
    fusion_model(cfg)


def system_transformation(cfg: Config, prefix: str = 'transformation_'):
    cfg[f'{prefix}input_size'] = cfg[f'{prefix}input_size'] \
        if f'{prefix}input_size' in cfg else cfg['system_embedding_size']
    cfg[f'{prefix}hidden_size'] = cfg[f'{prefix}hidden_size'] \
        if f'{prefix}hidden_size' in cfg else cfg['hidden_size']
    cfg[f'{prefix}hidden_layers'] = cfg[f'{prefix}hidden_layers'] \
        if f'{prefix}hidden_layers' in cfg else cfg['hidden_layers']
    cfg[f'{prefix}output_size'] = cfg[f'{prefix}output_size'] \
        if f'{prefix}output_size' in cfg else cfg['system_embedding_size']

    cfg[f'{prefix}model'] = cfg[f'{prefix}encoder'] if f'{prefix}encoder' in cfg else 'mlp'
    ARCH_CONFIG_REGISTRY[cfg[f'{prefix}model']](cfg, prefix)

    cfg['freeze_transformation'] = (cfg['freeze_transformation'] if cfg['freeze_transformation'] is not None
                                    else cfg['freeze']) \
        if 'freeze_transformation' in cfg else False


def system_decoder(cfg: Config, prefix: str = 'decoder_'):
    cfg['freeze_decoder'] = (cfg['freeze_decoder'] if cfg['freeze_decoder'] is not None else cfg['freeze']) \
        if 'freeze_decoder' in cfg else False

    defusion_model(cfg)
    process_decoder(cfg)


@register_model_architecture('system_model', 'system_model')
def system_model(cfg: Config):
    if cfg['process_autoencoder']:
        cfg['only_process'] = True
        cfg['freeze'] = True
        cfg['freeze_encoder'] = True
        cfg['freeze_decoder'] = True
        cfg['freeze_process_encoder'] = False
        cfg['freeze_process_decoder'] = False
        cfg['offset'] = 0

    if cfg['system_autoencoder']:
        cfg['only_system'] = True
        cfg['freeze'] = True
        cfg['freeze_encoder'] = False
        cfg['freeze_decoder'] = False
        cfg['freeze_process_encoder'] = True
        cfg['freeze_process_decoder'] = True
        cfg['load_process_encoder'] = True
        cfg['load_process_decoder'] = True
        cfg['offset'] = 0

    cfg['dropout'] = cfg['dropout'] if 'dropout' in cfg else 0.16

    cfg['input_size'] = cfg['jobs'] if 'jobs' in cfg else 16
    cfg['process_embedding_size'] = cfg['process_embedding_size'] if 'process_embedding_size' in cfg else 256
    cfg['system_embedding_size'] = cfg['system_embedding_size'] if 'system_embedding_size' in cfg else 256
    cfg['hidden_size'] = cfg['hidden_size'] if 'hidden_size' in cfg else cfg['system_embedding_size']
    cfg['hidden_layers'] = cfg['hidden_layers'] if 'hidden_layers' in cfg else 0
    cfg['output_size'] = cfg['output_size'] if 'output_size' in cfg else cfg['input_size']

    cfg['freeze'] = cfg['freeze'] or False if 'freeze' in cfg else False

    system_encoder(cfg, 'encoder_')
    system_transformation(cfg, 'transformation_')
    system_decoder(cfg, 'decoder_')
