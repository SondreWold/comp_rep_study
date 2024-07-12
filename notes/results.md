# Results 

## Base model training
Hyperparameters:
  - Hidden size: 512
  - Layers: 6
  - Dropout: 0.1
  - Gradient clip value: 0
  - Learning rate: 7e-5

Accuracy over seeds:
  - 1860: 91.04%
  - 3125: 91.06%
  - 4: 90.87%
  - 4421: 90.46%
  - 42: 90.11%

**Mean:90.708**,  **+- 0.3686**

## Base model evaluation on all the functions
{"base_tasks": [0.9063882316634091], "remove_second": [0.9944972486243121], "remove_first": [0.9804707060590886], "copy": [0.997], "append": [0.8739369684842421], "echo": [0.9855], "prepend": [0.904], "shift": [0.9785], "swap_first_last": [0.9925], "reverse": [0.977], "repeat": [0.9595]}
