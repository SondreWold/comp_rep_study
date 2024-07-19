# Meeting 19th of July 2024

Note from our meeting on 19th of July 2024.

## Immediate Next Steps
Design of experiments:

1) Overall functional capacity of functional subnetworks
   - Through pruning we obtain 17 subnetworks (1 pruned on base task + 10 atomic subtasks + 6 compositions of two atomic subtasks)
   - Performance overview of atomic subnetworks (11 x 11 on all atomic tasks + base task)
   - Performance overview of compositions. For each composition (6 times) we want a 3 x 3 plot (e.g. copy, append, copy_append)
  
2) Subnetwork set operations
    - Given 3x3 plots from above. We now extend the rows by new subnetworks acquired through set operations (let's define A: copy, B: append, C: copy_append. We add D: C - A, E: C - B, F: A + B, G: C - (A+B)).

3) Subnetwork overlaps
    - Given two subnetworks, e.g. C and (A+B). What is their IoU, etc.?  
