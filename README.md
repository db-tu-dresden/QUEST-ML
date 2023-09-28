# carQUEST-ML
Modeling queue simulations in production lines with ML

The usual steps are: *Data Creation*, *Training*, and finally *Inference*.

## Data Creation
First a dataset needs to be created. 
To enable this, create a directory with a config.yaml describing the configuration of the system 
([README.md](system/README.md) explains how this is done) 
and a graph_description.note describing the structure of the system 
([README.md](notation/README.md) explains how this is done).

When these two files are present in a directory run: 
```shell
python create_data.py --path PATH-TO-YOUR-DIR
```  
Optional arguments are:  
`--train N`, where `N` is the number of steps the simulation runs for the training data creation, the default is 10000.  
`--valid N`, where `N` is the number of steps the simulation runs for the validation data creation, the default is 2000.  
`--test N`, where `N` is the number of steps the simulation runs for the test data creation, the default is 10000.  

This creates the subdirectories `/train`, `/valid` and `/test`.
For each of these directories a separate simulation is run.  
Inside each directory a xarray DataArray is created containing all logged data.  
The dimensions of the DataArray are time (step), process, and job.  
Alongside, a plot is created displaying all process queues over the runtime of the simulation.  
In the given root directory an image is created describing the structure of the system.


## Training
To use the created data for training, simply run: 
```shell
python train.py --path PATH-TO-YOUR-DIR
```
This will start the training.  
Check the `ml` package [README](ml/README.md) for a detailed description of the arguments.

Some optional arguments are:  
`--gpu`, if set, the training uses GPUs, `--no-gpu` disables GPU usage. It is disabled by default.  
`--world_size N`, where `N` is the number of GPUs to use, default is all available.  
`--wandb`, enables wandb logging, `--no-wand` disables wandb logging. It is activated by default.  
`--save_model`, to save the model at the end of training, `--no-save_models` to explicitly don't save the model.

Use `--save_datasets` to save the created datasets.  
To reuse saved datasets, use `--load_datasets`.

Training includes preprocessing of the datasets by default.
If datasets are loaded using `--load_datasets` no preprocessing is done, the datasets are loaded as is.
Use `--only_preprocessing` to only run preprocessing.


## Hyperparameter search
For hyperparameter search [ray-tune](https://www.ray.io/ray-tune) is used. 
It is necessary to first create a dataset, see [Data creation](#data-creation).  
After this, simply run: 
```shell
python tune.py --path PATH-TO-YOUR-DIR
```  

Optional cli arguments are:  
`--samples`: This is the number of times to sample from the hyperparameter space, defaults to 10.  
`--max-epochs`: The maximal number of epochs to run per trail, defaults to 10.  
`--gpus`: GPUs to be used per trial, defaults to 0.  

The config describing the parameter search space is the dictionary `TUNE_CONFIG` defined in `ml/scripts/tune_config.py`. 
Its default value is:
```python
{
    'hidden_size': tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    'hidden_layers': tune.sample_from(lambda _: np.random.randint(2, 4)),
    'learning_rate': tune.loguniform(1e-4, 1e-1),
    'batch_size': tune.choice([2, 4, 8, 16])
}
```

To tune nuisance parameters while tuning a hyperparameter use the script [do_tune.sh](./do_tune.sh). 
Adapt the script by changing the variables. The variable `list` holds the sequence of values for the hyperparameter
to be tuned. Adapt the `TUNE_CONFIG` in the string to specify the nuisance parameters.

## Inference
Use this to get to a certain target with a pretrained model. 

```shell
python infer.py --path PATH-TO-YOUR-DIR [ STEP_TO_TARGET | STEP_UNTIL ]
```  

Like in training, the model and architecture need to be described with `--arch` and possibly additional arguments to 
set model/architecture parameters.
The architecture will be created and loaded from the model save path, if not specified this is within a directory 
called `save` in the path given by `--path`.
The file name is the architecture name followed by `_model.pt`.
An example is `./save/-/save/FlatMLP_model.pt`.  
The model save path can also be set manually with the CLI argument `--model_save_path`.
Important is that the specified architecture match the saved model.

The provided path also needs to include a file called `inference_config.yaml`, 
including the entries `initialState` and `targetDist`.

Optional CLI arguments are:  

`--k_model`: This is the number of times the model based simulation is runs from the initial state, default is 1.  
`--k_simulation`: This is the number of times the simulation runs from the initial state, default is 1.  
`--mutate`: This is relevant if `k_model` is > 1. For each additional run the initial state is mutated 
by adding an integer tensor sampled from a uniform distribution to the initial state. 
It is active by default if `k_model` is larger 1.  
`--mutation_low`: Controls the lower bound of the uniform distribution used by `mutate`. Default is -2.  
`--mutation_high`: Controls the upper bound of the uniform distribution used by `mutate`. Default is 2.  
`--job_arrival_path`: Path to a yaml file containing job arrivals. 
See [System Readme](./system/README.md) for further detail.  
`--verbose`: If set the final state of each simulation run will be printed.

The script has two actions, `STEP_TO_TARGET` and `STEP_UNTIL` one of them must be chosen.  

`STEP_TO_TARGET`: used for reaching a target distribution from some initial state.  
Arguments are:  
`--limit`: Maximal number of steps the model should take.

`STEP_UNTIL`: used for stepping N times with the model.  
Argument is:  
`--until`: Number of steps to take.

After the model ran `k_model` times , the simulation is run from the initial state `k_simulation` times.
Mean and standard deviation of all final states of the simulation are printed, in addition to 
the closest matching final state from the model inference.
