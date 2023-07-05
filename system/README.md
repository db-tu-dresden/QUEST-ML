# System

## Getting started

Import the following classes:
```python
from system import Config, Environment, System
```

Load the config:
```python
config = Config('config.yaml')
```
The config is further explained in section [Config](#config). 
Note that the job names specified in the config need to be identical to the data elements in the notation.


Create an instance of the simulation environment:
```python
env = Environment()
```
The `Environment` class is just a wrapper of the `simpy.Environment`.

Create a system instance:
```python
system = System(config, notation, env=env)
```
The `System` needs the aforementioned config and environment, it also relies on a notation instance.
For information how the notation is created please read the [README.md](../notation/README.md) of the notation package. 
Note that the arrival process, the process creating the jobs, needs to be explicitly specified. 
It is the first node of any given notation.

Then build and run the simulation with:
```python
system.build()
system.run()
```

The jobs of each process over time can be plotted by:
```python
system.logger.plot()
```

The job distribution of each proces over time can be saved in a pandas dataframe by:
```python
system.logger.save_df('./save/df.pkl')
```

## Config
The following is an example config.
```yaml
until: 10000                      # number of steps the simulation will run
loggingRate: 0.1                  # rate at which the logger will log. 0.1 means it will log 10 time per step
randomSeed: 42                    # seed for initializing the numpy.SeedSequence from which the random number generators are created
jobs:                             # list describing all jobs
  - name: 'A'
    arrivalProbability: 0.1
    failureRate: 0
  - name: 'B'                     # name of the job
    arrivalProbability: 0.1       # probability with which a created job is of this job type. Needs to sum to 1.0 over all jobs
    failureRate: 0                # probability with which the job might fail quality control 
  - name: 'C'
    arrivalProbability: 0.8
    failureRate: 0
arrivalProcess:
  beta: 1.0                       # this is the beta value of the exponential function used for simulating the job arrival time
processes:                        # describes the processing time drawn from the normal distribution of a job
  mean: 1.0                       # mean value for the processing time
  std: 0.2                        # std deviation of the processing time
```