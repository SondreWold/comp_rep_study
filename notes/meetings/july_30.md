# July 30

# Experiment 1

### Motivation

Atomic circuits can solve their own test set. What about their unions? Can these combined models solve the test sets of the individual circuits used in the combination?

### Hypothesis
Yes.

### Method

| Model | Copy | Echo | Copy_Echo |
|----------|----------|----------| ----------|
| copy | 1.0 |  0.0970 | 0.5480  |
| echo | 0.0 | 1.0 | 0.5 |
| copy_echo | 1.0 | 1.0 |  1.0 |
| copy $\cup$ echo |  0.0 | 0.0783 | 0.0383 |
| echo $\cup$ copy |  0.0 | 0.0783 | 0.0383 |
| echo $\cup$ copy $\cup$ copy_echo| 0.0 | 0.4285 | 0.2075|

**Note:** We operated on all the layers and both for linear and layernorm modules.

### Findings

This does not seem to be the case. It is more complicated than initially thought. 

This **might** indicate that circuits interfere, so that the addition of one circuit to another one, messes things up unless this addition is "cancelled" out by something else. 

What is this _something else_?

# Experiment 2

### Motivation

The findings from experiment 1. Why and what is interferring?

### Hypothesis

Models learn circuits that cancel eachother out in order to solve individual atomic functions.

### Method

Remove copy from the base_model and evaluate on all atomic tasks.

Consider the circuit $\hat{E}=\bigcap_i f_i$.

Now do:

$\hat{T} = M - C + \hat{E}$, with $M$ being the base model.

If $\hat{T}$ now can still do all the atomic tasks, except $C$, then the network is compositional.

If $\hat{T}$ now fails for other tasks than $C$, then this supports the cancel hypothesis. 

### Findings:

$\hat{T} = M - C + \hat{E}$ fails on all tasks.

$f_i - \hat{E}$ also fails on all tasks.

This indicate that the base model has not learned a compositional solution to the problem. 

# Experiment 3

### Motivation

Why is the difference operation destroying the performance completely? Even where there are super sparse circuits?

Removing the xtremely sparse echo circuit from the base model, which can solve all 10 individual tasks almost perfectly, results in a model that can't do anything, even if we add back the "glue".

### Hypothesis

Each circuit contains something that is unique to that circuit but which is also crucial for solving all the other tasks.

### Method

This "something" might not be distributed equally across the layers. What happens if we only take the difference layer and component wise?

$M$ and $echo$ has some overlap. Specifically:
- In all layers for both the encoder and decoder, there is little sharing between the linear layers, which are also extremely sparse.
![base_echo_fraction](./figures/base_VS_echo_fraction.png)


**Removing both linear and layernorms per layer:**

| Layer(s) removed | Accuracy on copy |
|----------|----------|
| 0 | 0.0 |
| 1 | 0.375 |
| 2 | 0.968 |
| 3 | 1.0 |
| 4 | 0.968 |
| 5 | 1.0 |
| ---- | ---- |
| 1, 2, 3, 4, 5 | 0.0 |
| 2, 3, 4, 5 | 0.145 |
| 3, 4, 5 | 0.56 |
| 4, 5 | 0.687 |
| 2, 3, 4 | 0.70 |
| 3, 4 | 0.976 |


**Removing just the linear**

| Layer | Accuracy on copy |
|----------|----------|
| 0 | 0.687 |
| 1 | 0.468 |
| 2 | 1.0 |
| 3 | 1.0 |
| 4 | 1.0 |
| 5 | 1.0 |

**Removing just the norms**

| Layer | Accuracy on copy |
|----------|----------|
| 0 | 0.0 |
| 1 | 0.5 |
| 2 | 0.968 |
| 3 | 1.0 |
| 4 | 1.0 |
| 5 | 1.0 |


**$M - echo$: Removing layers and norms and evaluating across tasks**

| Layer(s) | copy | echo | repeat | 
|----------|----------| ----------| ----------|
| base_model | 1.0 | 1.0 | 0.983 |
| 0 | 0.0 | 0.0 | 0.0 |
| 3, 4 | 0.976 | 0.835 | 0.8 |

**$M - copy$: Removing layers and norms and evaluating across tasks**

| Layer(s) | copy | echo | repeat | 
|----------|----------| ----------| ----------|
| base_model | 1.0 | 1.0 | 0.983 |
| 0 | 0.07 | 0.015 | 0.0 |
| 3, 4 | 0.976 | 0.898 | 0.82 |


### Findings

It is clear that the first two layers are the most crucial to the performance, even when you consider the fact the these are also the most shared. Removing layers 2, 3, 4, and 5 from $M$ by the difference with $echo$, is less deterimental to the performance than removing just the first layer.
Similarily, the final layer is also important, or put in another way: the middle layers do not do much: removing layer 2, 3 and 4 still gets a performance of 70%. 

Also: removing the third and fourth layer of echo from the base model hurts performance over multiple tasks.

# Experiment 4

### Motivation

Is $C$ the only subspace of $M$ than can solve $c$? Can we find another subspace of $M$ that does $c$? This might also explain some of the results we are seeing.

Our masks "picks out" a set of the original parameter space that can solve a task, e.g it finds a combination of parameters C that can approximate the same linear transformation as the original. It does not follow from this that C is the only set of parameters that can solve the same task.

A neural network is a learned transformation that maps inputs to outputs $F_\theta: x \rightarrow y$. If we can find multiple sparse masks over $\theta$ that can do this transformation, then the mask is essentially learning a sparse approximation of the function. This approximation does not have to be unique, which questions the idea of a circuit. 

Idea: Can we find another C circuit that is different than C? We can artificailly inhibit C, and then try to find C prime?


### Method

Train three different masks for $copy$ with the same regularization but with different seeds and learning rates and initial mask value and see to which extent these models overlap (and if the all solve the task).

### Findings
