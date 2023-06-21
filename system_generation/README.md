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
Anchors are also part of the notation, they can be used to reference an existing node.
The notation string has two main parts, the base part and the appendix.  
The base part describes the initial structure. 
Any references used must be described in the appendix.
Base and appendix are seperated by a new line (```\n```).

The appendix can contain multiple lines seperated by a new line.
Each line describes one reference. 
It consists of a reference key followed by a colon and the description of the subgraph in valid notation.


#### Line
A line segment is represented by ```-```, it connects two nodes.  
A line is composed of one or multiple line segments. It can be writen as:  
```-```: one line,  
```--```: two lines,  
```-n-```: n + 2 lines.

#### Fork
A fork connects one node to multiple sub graphs and joins them together in the end.  
It is represented by ```<E>```, where ```E``` is a notation element (Line,  Fork or a combination of both).  
```E``` represents both the upper and lower path in the graph. A fork in this style thereby is symmetric.  

A fork can also make use of references. This is done by using bracket notation ```<[$1, $2, ...]>```,  
where ```$1``` references a subgraph. Forks written in bracket notation must include at least two references.

#### References
References enable the usage of sub graphs, as well as multiple and asymmetric paths in a fork.  
Denoted by ```$N```, where ```N``` is some number. 
A used reference must be defined in the appendix.  
References can also use other references.  
Note: there is no check for cyclic references; this is expected from the user.

### Anchors
Anchors are denoted by ```!N``,` where ```N``` is some number.  
They are a notation element and are used as reference for a node, this enables multiple paths to end at one specific node.

#### Examples
Simple examples: ```--```, ```-<-1->-```, ```-<<>>-```, ```-<-<-1->->-```, ```<->```.  

Examples with references:  
```-<[$1, $2]>-\n$1: -\n$2: -```,  
```-<[$1, $2, $3]>-\n$1: -\n$2: -\n$3: -```,  
```-<[$1, $2]>-\n$1: -<>-\n$2: <>```,  
```-<[$1, $2]>-\n$1: -<[$3, $2]>-\n$2: <>\n$3: -```.

Examples with anchors:  
```
-<[$1, $2]>-\n
$1: <[$3, $4]\n
$2: !2\n
$3: !1\n
$4: !2
```

```
-<[$1, $2]>\n
$1: -<[$3, $4]\n
$2: -!1\n
$3: --!2\n
$4: !1
```

