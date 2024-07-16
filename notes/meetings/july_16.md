# Meeting 16th of July 2024
Note from our meeting on 16th of July 2024.

## Immediate Next Steps

Implement weight set operations for comparison of trained masks.

Csordas (2020) uses two metrics:
- Intersection over Union
  - "...measures how much the weights used for solving the tasks overlap".
- Intersection over Minimum
  - "...measures the number of overlapping weights divided by the minimum of the total number of weights used for each task".
  - A measure of "subsetness"
- If all weights are shared, both IoU and IoMin are 1, but if the weights needed for one task are a strict subset of the other the IoMin is 1 while IoU < 1.
- They measure both per layer on a FF and a LSTM, but they do not do it on the Transformer.
