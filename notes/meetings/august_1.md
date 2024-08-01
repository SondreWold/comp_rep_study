# August 1
Philipp and Sondre


## Where we are
- No compositionality given the identified circuits
  - Evidence:
    - Experiment 1 and 2
- Each circuit contains something that is unique to that circuit but which is also crucial for solving all the other tasks.
  - The intersection of all the models, $\hat{E}$ was not enough to recover the performance
  - Removing copy from the base model destroys the performance on everything, even with E hat. 

## Open questions
- Why is $\hat{E}$ not the whole story of the glue.
- Are there multiple circuits in M that can solve the same task?
  If so, the network is redundant. 
- Is there some fundamental text processing in the lower layers that is not captured by $\hat{E}$ and that is different across the circuits?
  - Related to experiment 3.
- Cancellation hypothesis.
- Everything is a transformation, so removing or adding anything projects the input into another direction in the activation space. 

## Going forward
- Create a new $M$ model that is the original base model but with a b_matrix that has all 1's and redo all the experiments. 
- A circuit is just a sparse approximation of M for a subset of the data. So:
  - If $x_c:M \rightarrow y_c$ and $x_c:C \rightarrow y_c$, then removing C from M destroys the transformation. Adding C, and even $\hat{E}$, does not that recover that.
  - Now if also $x_e:M \rightarrow y_e$ and $x_e:E \rightarrow y_e$
  - We know that $x_{ec}:M \rightarrow y_{ec}$, so can we combine C and E to construct the circuit X that does $x_{ec}:X \rightarrow y_{ec}$? Or: is the CE circuit a linear combination of C and E?
  - Our plan for testing this is to per layer $l$:
    - Solve the linear combination: $\alpha \times C_{bmatrix}^l + \beta \times E_{bmatrix}^l = CE_{bmatrix}^l $
  - We can also take the union of C and E and then prune on CE, only allowing removal of items, not adding new ones. This would be a learned global solution to the overall mapping. If you cant do this, then there is no linear combination, we think. If it is, then the composition is way more complicated than a mask union. 

# Timeline
- Next week we do this
- Then two weeks break
- We pick up in the end of august.
