# System Generation

## Random Generation
Simple instructions for random system generation can be given in [config.yaml](config.yaml).
If follows the schema provided in [schema.yaml](schema.yaml).
An example config is:
```yaml
jobCount: 3
branchingFactor:
  min: 1    # default 1
  max: 2    # default 1
length: 4
processCount: 6
```
```jobCount``` is the total number of jobs.  
```branchingFactor``` describes the range of possible branches after a process.  
```length``` is the length of the shortest path from start to finish in the system.  
```processCount``` is the number of processes in the system. It can not be lower than ```length```.

Note, only ```jobCount``` is required.