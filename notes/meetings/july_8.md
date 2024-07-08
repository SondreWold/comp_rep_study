# Meeting 8th of July 2024
Note from our meeting on 8th of July 2024.

## Immediate Next Steps
The following steps should be done next:
- Ensure that data generation scripts are available
- Evaluate our trained masks (`ContinuousMask` and `SampledMask`)
  - First validate the initial frozen model on the `Append` validation set (get accuracy)
  - Validate mask training on whole dataset
  - Train maskedmodel on `Append` training set (validate during training), make sure that train and val accuracy go up, while number of remaining weights goes down
  - Do hyperparam search (Continuous: learning rate, parameter initializiations, Sampled: alpha, learning rate, num steps)
  - Repeat evaluation for all 10 functions
  - Repeat evaluation for both methods
- Compare different masks obtained via different pruning techniques
  - Employ comparison metric of Schmidthuber

## Longer-Term Goals
The long-term steps:
- Finish model pruning
- Start implementing SAE
