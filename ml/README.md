# Machine Learning Module

## Usage

To use the module import the following:
```python
from ml import Config, Trainer, Parser
from ml.models import build_model
```



`Parser` is a subclass from `argparse.ArgumentParser` containing the relevant arguments.  
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

Create a parser based on the config values with
```python
parser = Parser(config)
```
The provided config supplies the argument names and default values to the parser.  
Every single value can then be changed by prepending `--`, e.g. set the hidden size with `--hidden_size 8`.  
The only argument differing in name to the respective entry in the config is `base_path`.  
This can be shortend to `--path MY-BASE_PATH`.  
Provide a path to a directory where the respective `DataArrays` can be found.  
The directory should contain a directory called `data`.  
This then should contain three directories called `train`, `valid`, and `test`, 
where each contains a respective `da.pkl` file.

Optionally, the datasets can be provided directly to the `Trainer` instance 
by passing them as arguments to the `run` methode.

Parse the arguments like usual
```python
args = parser.parse_args()
```

To update the config with the arguments run
```python
config.update_from_args(args)
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
The arguments are all entries in the config file provided to `Config` (see [config.yaml](config.yaml)).  
To set boolean values prepend `no` to set to false,  
e.g. for GPU usage, use `--gpu` to set it to `True` and `--no-gpu` to set it to `False`.  
Additionally, the `base_path` can be set with the shorter argument name `--path`.

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