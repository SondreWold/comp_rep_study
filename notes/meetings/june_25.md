# Meeting, 25th of June

Who: Robert, Philipp and Sondre

## Next steps

- We are going to train small transformer models on PCFG SET using
  different methods
- Our first concrete research question would be in the line of: "Is the
  subnetwork for a higher-order function like repeat a superset of the
  subnetwork for copy?"
- What we need:
    - 10 subnetworks using **m** methods
    - **10 x m** function-method matrix that we can compare
    - A comparison method for comparing the methods, eg:
        - Cosine
        - PCA + Cosine
        - Indicie based overlap
        - Layer by layer
        - Isotropy
