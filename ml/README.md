# Machine Learning Module

## Usage

To use the module import the following:
```python
from ml import Config, Trainer, parser
from ml.models import build_model, parse_arch
```

`parser` is an `argparse.ArgumentParser` instance containing the relevant arguments.  
It is explained in more detail in section [Parser](#parser).  
Additional arguments can be added to the parser as usual.

To retrieve the arguments, use the provided `parse_arch` function, e.g. 
```python
args = parse_arch(parser)
```

Load the config by providing a location to the respective `.yaml` file.
```python
config = Config('ml/config.yaml')
```

Provide a path to a directory where the respective `DataArrays` can be found.  
The directory should contain a directory called `data`.  
This then should contain three directories called `train`, `valid`, and `test`, 
where each contains a respective `da.pkl` file.  
The base path can be set as follows:
```python
config.set_base_path(base_path)
```

Optionally, the datasets can be provided directly to the `Trainer` instance 
by passing them as arguments to the `run` methode.

To pass the cli arguments to the config run:
```python
config.add_from_args(args)
```

To build the model use:
```python
model = build_model(config)
```

Finally, to run the training execute:
```python
Trainer.run(config, model)
```


## Parser
To display all possible parser arguments use the `--help` argument.
Possible arguments are:  
`--gpu`, whether to use GPUs, default `False`.  
`--n_gpu`, how many GPUs to use, default all present.  
`--wandb`, whether to use wandb for logging, default `True`.  
`--arch`, what architecture to use, required.

To display all available arguments to a specific architecture use `--arch SOME_REGISTERED_ARCHITECTURE --help`.


## Models
Models can be added. They need to extend the `ml.models.Model` class.  
They need to override the following methods:  
```python
@staticmethod
def add_args(parser: argparse.ArgumentParser):
    pass                # add arguments needed for the model
```

```python
@classmethod
def build_model(cls, cfg: Config):
    pass                # build the model given the config and return an instance
```

```python
def forward(x):
    pass                # do the forward pass
```

To register a model use the decorator `register_model` by providing a model name, like:
```python
@register_model('custom_model')
class CustomModel(Model):
    pass
```

To register an architecture use the decorator `register_model_architecture` by providing a model name and an architecture name:
```python
@register_model_architecture('custom_model', 'custom_model_2_layer_30_hidden')
def custom_model_2_layer_30_hidden(cfg: Config):
    pass
```
In this function set the needed parameters to the config.