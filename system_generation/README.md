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

## Notation
The notation describes the structure of a very simple planar graph.  
The notation is composed out of lines and forks representing the edges of the graph, nodes are implicit.

#### Line
A line segment is represented by ```-```, it connects two nodes.  
A line is composed of one or multiple line segments. It can be writen as:  
```-```: one line,  
```--```: two lines,  
```-n-```: n + 2 lines.

#### Fork
A fork connects one node to exactly two sub graphs.
It is represented by ```<E>```, where ```E``` is a notation element (Line,  Fork or a combination of both).
```E``` represents both the upper and lower path in the graph. A fork thereby is symmetric.

#### Example
A valid notation examples are: ```--```, ```-<-1->-```, ```-<<>>-```, ```-<-<-1->->-```, ```<->```.
