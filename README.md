# carQUEST-ML
Modeling queue simulations in production lines with ML

## Data creation
First a dataset needs to be created. 
To enable this, create a directory with a config.yaml describing the configuration of the system 
([README.md](system/README.md) explains how this is done) and a graph_description.note describing 
the structure of the system ([README.md](notation/README.md) explains how this is done).

When these two files are present in a directory run: `python create_data.py --path PATH-TO-YOUR-DIR`.
This creates the subdirectories `train`, `valid` and `test`.
For each of these directories a separate simulation is run.
Inside each directory a pandas dataframe is created containing all logged data. 
Alongside, a plot is created displaying all process queues over the runtime of the simulation.
In the given root directory an image is created describing the structure of the system.


## Training
To use the created data for training, simply run: `python train.py --path PATH-TO-YOUR-DIR`.
This will start the training.
If the ml config value `wandb` is `True` and a user is logged into wandb, 
training metrics can be seen on the given website.
At the end of the training the model is saved.


## Hyperparameter search
For hyperparameter search [ray-tune](https://www.ray.io/ray-tune) is used. 
It is necessary to first create a dataset, see [Data creation](#data-creation).  
After this, simply run: `python raytune.py --path PATH-TO-YOUR-DIR`.  

Optional cli arguments are:  
`--samples`: This is the number of times to sample from the hyperparameter space, defaults to 10.  
`--max-epochs`: The maximal number of epochs to run per trail, defaults to 10.  
`--gpus`: GPUs to be used per trial, defaults to 0.  

The config describing the parameter search space is the dictionary `TUNE_CONFIG` defined in `raytune.py`. 
Its default value is:
```python
{
    'hidden_size': tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    'layers': tune.sample_from(lambda _: np.random.randint(2, 4)),
    'learning_rate': tune.loguniform(1e-4, 1e-1),
    'batch_size': tune.choice([2, 4, 8, 16])
}
```