import argparse
import importlib
import os
import typing

from ml import Config
from ml.models.base import Model, DistributedDataParallel

MODEL_REGISTRY = {}
ARCH_MODEL_REGISTRY = {}
ARCH_MODEL_NAME_REGISTRY = {}
ARCH_MODEL_INV_REGISTRY = {}
ARCH_CONFIG_REGISTRY = {}


def get_model_from_type(model_type: str, config: Config) -> typing.Type[Model]:
    model = ARCH_MODEL_REGISTRY[model_type]
    ARCH_CONFIG_REGISTRY[model_type](config)

    assert model is not None, (
        f'Could not infer model type from {config}. '
        f'Available models: {MODEL_REGISTRY.keys()}'
        f'Requested model type: {model_type}'
    )
    return model


def build_model(config: Config) -> Model:
    model_type = config['arch'] if 'arch' in config else None
    model = get_model_from_type(model_type, config)
    return model.build_model(config)


def register_model(name: str) -> typing.Callable:
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('mlp')
        class MLP(Model):
            (...)

    .. note:: All models must implement the :class:`Model` interface.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls: typing.Type[Model]):
        if name in MODEL_REGISTRY:
            raise ValueError('Cannot register duplicate model ({})'.format(name))
        if not issubclass(cls, Model):
            raise ValueError(
                'Model ({}: {}) must extend BaseFairseqModel'.format(name, cls.__name__)
            )
        MODEL_REGISTRY[name] = cls

        return cls

    return register_model_cls


def register_model_architecture(model_name: str, arch_name: str) -> typing.Callable:
    """
    New model architectures can be added to fairseq with the
    :func:`register_model_architecture` function decorator. After registration,
    model architectures can be selected with the ``--arch`` command-line
    argument.

    For example::

        @register_model_architecture('mlp', 'mlp_10_layers')
        def mlp_10_layers(cfg):
            cfg['layers'] = cfg['layers'] if 'layers' in cfg else 10
            (...)

    The decorated function should take a single argument *cfg*, which is a
    :class:`ml.Config`. The decorated function should modify these
    arguments in-place to match the desired architecture.

    Args:
        model_name (str): the name of the Model (Model must already be
            registered)
        arch_name (str): the name of the model architecture (``--arch``)
    """

    def register_model_arch_fn(fn: typing.Callable) -> typing.Callable:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                'Cannot register model architecture for unknown model type ({})'.format(
                    model_name
                )
            )
        if arch_name in ARCH_MODEL_REGISTRY:
            raise ValueError(
                'Cannot register duplicate model architecture ({})'.format(arch_name)
            )
        if not callable(fn):
            raise ValueError(
                'Model architecture must be callable ({})'.format(arch_name)
            )
        ARCH_MODEL_REGISTRY[arch_name] = MODEL_REGISTRY[model_name]
        ARCH_MODEL_NAME_REGISTRY[arch_name] = model_name
        ARCH_MODEL_INV_REGISTRY.setdefault(model_name, []).append(arch_name)
        ARCH_CONFIG_REGISTRY[arch_name] = fn
        return fn

    return register_model_arch_fn


def import_models(models_dir, namespace: str):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
        ):
            model_name = file[: file.find('.py')] if file.endswith('.py') else file
            importlib.import_module(namespace + '.' + model_name)

            if model_name in MODEL_REGISTRY:
                parser = argparse.ArgumentParser(add_help=False)
                group_archs = parser.add_argument_group('Named architectures')
                group_archs.add_argument(
                    '--arch', choices=ARCH_MODEL_INV_REGISTRY[model_name]
                )
                group_args = parser.add_argument_group(
                    'Additional command-line arguments'
                )
                MODEL_REGISTRY[model_name].add_args(group_args)
                globals()[model_name + '_parser'] = parser


def add_arch_args(parser: argparse.ArgumentParser, key: str, title: str = None, prefix: str = '', default=None):
    args, _ = parser.parse_known_args()

    if default is not None and not hasattr(args, key):
        setattr(args, key, default)

    # Add model-specific args to parser.
    if hasattr(args, key):
        arch_name = getattr(args, key)
        model_specific_group = parser.add_argument_group(
            title,
            # Only include attributes which are explicitly given as command-line
            # arguments or which have default values.
            argument_default=argparse.SUPPRESS,
        )

        model_specific_group.root_parser = getattr(parser, 'root_parser', parser)

        if arch_name in ARCH_MODEL_REGISTRY:
            ARCH_MODEL_REGISTRY[arch_name].add_args(model_specific_group, prefix=prefix)
        elif arch_name in MODEL_REGISTRY:
            MODEL_REGISTRY[arch_name].add_args(model_specific_group, prefix=prefix)
        else:
            raise RuntimeError()


def parse_arch(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args, _ = parser.parse_known_args()
    add_arch_args(parser, 'arch', 'Model-specific configuration')
    return parser.parse_args()


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_models(models_dir, 'ml.models')
