# Revealing-Graph-Structures

**Code**: 
## Inputs:

It takes two files as input, the graph file and the lookup file used to indicate which nodes to embed.

### Input graph file
The input graph file is the static edge list in the following format separated by tab:
```
<src> <dst> <weight>
```
The edge list is assumed to be re-ordered consecutively from 0, i.e., the minimum node ID is 0, and the maximum node ID is <#node - 1>. A toy static graph is under "/graph/" directory.

### Input lookup file
The lookup file is a subset of nodes in the graph to embed. A toy example is under "/graph/" directory.

### Other input arguments
## The specific configuration to pass can be found with the `python main -h` command.

