# Meeting 8th of July 2024
Note from our meeting on 8th of July 2024.

## Immediate Next Steps
The following steps should be done next:
- Ensure that data generation scripts are available
- Evaluate our trained masks (`ContinuousMask` and `SampledMask`)
  - Train an initial base model with hyperparameters from Sondre (across 5 random seeds), report average +- stdev on base task (Sondre)
  - Evaluate base model on all subtasks (Sondre)
  - Share best base model (Sondre)
  - Run sweeps for all subtasks and base task on `ContinuousMask`, thus obtaining 11 pruned models (Philipp)
  - Evaluate all 11 pruned models on all 10 subtasks + base task (Philipp)
- Repeat sweep, pruning and evaluation for `SampledMask`
- Implement ways to compare subnetworks
  - Employ comparison metric of Schmidthuber
- Implement ways to operate (union, intersection, inverse, difference) on subnetworks

## Longer-Term Goals
The long-term steps:
- Finish model pruning
- Start implementing SAE
