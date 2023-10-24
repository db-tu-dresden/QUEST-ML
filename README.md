# QUEST-ML
Queuing System Simulation Modeling with Machine Learning

## Quick Start

Look in the directory `graphs/` and pick a graph you like to continue with. 
Use the path to the directory in the following examples. 
For simplicity `./graphs/-` is used in all examples in this section.

**First**, install all necessary python packages with 
```shell
pip install -r requirements.txt
```

**Second**, run data creation with
```shell
python create_data.py --path ./graphs/-
```

**Third**, train on the created data with
```shell
python train.py --path ./graphs/- --arch flat_mlp --epochs 10 --accumulation_window 100
```

**Fourth**, run predictions with
```shell
python predict.py --path ./graphs/- --arch flat_mlp --method method1
```

## Data Creation
First a dataset needs to be created. 
To enable this, create a directory with a `config.yaml` describing the configuration of the system 
([system/README.md](system/README.md) explains how this is done) 
and a `graph_description.note` describing the structure of the system 
([notation/README.md](notation/README.md) explains how this is done).

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
Check the `ml` package [ml/README.md](ml/README.md) for a detailed description of the arguments.

Some optional arguments are:  
`--gpu`, if set, the training uses GPUs, `--no-gpu` disables GPU usage. It is disabled by default.  
`--world_size N`, where `N` is the number of GPUs to use, default is all available.  
`--wandb`, enables wandb logging, `--no-wand` disables wandb logging. It is activated by default.  
`--save_model`, to save the model at the end of training, `--no-save_models` to explicitly don't save the model.

Note, the CLI arguments are used to override the default values set in [ml/config.yaml](ml/config.yaml), 
you may also override the config to match your expected default values.

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
`--samples`: integer; number of times to sample from the hyperparameter space, defaults to 10.  
`--max-epochs`: integer; maximal number of epochs to run per trail, defaults to 10.  
`--gpus`: float; GPUs to be used per trial, defaults to 0.  

The config describing the parameter search space is the dictionary `TUNE_CONFIG` defined in 
[ml/scripts/tune_config.py](ml/scripts/tune_config.py). 
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
An example is `./graphs/-/save/FlatMLP_model.pt`. 
The model save path can also be set manually with the CLI argument `--model_save_path`.
Important is that the specified architecture match the saved model.

The provided path also needs to include a file called `inference_config.yaml`, 
it has the entries `initialState` and `targetDist`.

Optional CLI arguments are:  

`--k_model`: This is the number of times the model based simulation is runs from the initial state, default is 1.  
`--k_simulation`: This is the number of times the simulation runs from the initial state, default is 1.  
`--mutate`: This is relevant if `k_model` is > 1. For each additional run the initial state is mutated 
by adding an integer tensor sampled from a uniform distribution to the initial state. 
It is active by default if `k_model` is larger 1.  
`--mutation_low`: Controls the lower bound of the uniform distribution used by `mutate`. Default is -2.  
`--mutation_high`: Controls the upper bound of the uniform distribution used by `mutate`. Default is 2.  
`--job_arrival_path`: Path to a yaml file containing job arrivals. 
See [system/README.md](system/README.md) for further detail.  
`--verbose`: If set the final state of each simulation run will be printed.

The script has two actions, `STEP_TO_TARGET` and `STEP_UNTIL`, one of them must be chosen.  

`STEP_TO_TARGET`: used for reaching a target distribution from some initial state.  
Arguments are:  
`--limit`: Maximal number of steps the model should take.

`STEP_UNTIL`: used for stepping N times with the model.  
Argument is:  
`--until`: Number of steps to take.

After the model ran `k_model` times , the simulation is run from the initial state `k_simulation` times.
Mean and standard deviation of all final states of the simulation are printed, in addition to 
the closest matching final state from the model inference.


# Prediction
Use this to get the sequence of jobs that need to be put into the system to reach some target output distribution 
from an initial state.

```shell
python predict.py --path ./graphs/- --arch flat_mlp --method [ method1 | method2 | method3 ]
```

The initial state and target distribution need to be provided inside a `inference_config.yaml` file, 
see [graphs/-/inference_config.yaml](graphs/-/inference_config.yaml).
This file needs to be inside the directory provided by the `--path` argument.

The primary argument for this script is `--method`, it states the method that should be used for 
reaching the target distribution from the initial state.  
These methods are:  

`method1`: Running the simulation from the initial state until the target state is reached. 
This is done multiple times to find the best simulation run.  

`method2`: Using the trained model to step from the initial state until the target state is reached. 
Note, this requires the model to be trained with a POSITIVE offset, the default offset is `1`. 
The offset describes in which direction the model steps and how large the step is. 
Also note that one model step is dependent on the logging rate of the simulation and the scaling factor of the dataset.
So one model `step` is `logging_rate * scaling_factor`, default `logging_rate` is `0.1` and default `scaling_factor` is `1`.

`method3`: This tries to reduce the search space of the simulation with the help of the pretrained model. 
The model is used to run backwards from a target state that includes the target distribution 
until the initial state is reached. 
This results in an initial sequence of jobs to be added. 
The simulation then runs multiple times from the initial state to the target state, utilizing the sequence of jobs provided.
Between each simulation run the sequence of jobs is altered in the sense that some jobs are added and some are removed.
Note, this requires the model to be trained with a NEGATIVE offset, e.g. provide `--offset -1` to the training command.


Optional CLI arguments are:

`--max_model_steps`: Maximum steps for the model to take, default `200`. 
This is highly dependent on the desired target distribution.
Note that one model step does NOT necessarily equate to one simulation step.
The model step size is dependent on the config values of `scaling_factor`
and `logging_rate`, the model step size is `scaling_factor * logging_rate`!

`--max_model_simulations`: Used in method3. Maximum times for simulation to run, default `1`.

`--max_simulations`: Maximum times for simulation to run, default `100`.

`--max_simulation_steps`: Maximum steps for the simulation to take, default `20`. 
This value should be highly dependent on the desired target distribution. 
Generally the lower, the better, to finish off simulations that run too long.

`--mutations`: Number of times the original job arrivals are mutated, default `30`.

`--sub_mutations`: Number of times the original mutated job arrivals are subsequently mutated, default `1`.

`--mutation_low`: Low value of the uniform probability distribution used for mutation, default `-3`.

`--mutation_high`: High value of the uniform probability distribution used for mutation, default `3`.

`--print_quickest`: Number of quickest elements that are printed out, default `3`.

`--job_arrival_save_dir`: Directory where the created job arrival sequences are saved. 
Defaults to `dir_given_by_path_arg/job_arrivals`.