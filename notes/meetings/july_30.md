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

