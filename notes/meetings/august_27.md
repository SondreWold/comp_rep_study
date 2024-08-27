# Meeting, August 27.
Philipp and Sondre.

### General Discussion
Given our observations of previous experiments, we find that the pruning methods proposed by previous works are not suitable for circuit
identification (we should report this).
Given this and the fact that there exist a lot of different circuit identification methods with no unified framework, we decide to do a
comparative study in which we analyze and compare the circuits found by these different methods.

General questions that might be worth discussing (on a higher level):
- Can we identify a circuit (an isolated entity/module of the model) that can solve a certain subtask (of PCFGs)?
- Is this circuit unique -> this is impossible to answer, instead we will focus on: how do different circuits identified relate?
- We need to decide on metrics that we can use to compare these circuits (IoU, indirect effect, etc.)

### Immediate next steps (27.08 - 31.08)
- Define OUR meaning of circuit (that we look into within the paper)
- Both of us look into circuit identification methods and create a list of suitable methods to compare
- We select appropriate candidates from the union of that lists
