# Downloading the data

The data consists of two parts which can be downloaded and processed
individually.

### Base data

The base data includes the original data splits from the PCFGS paper which
we transform to a ; seperated file.

Run:

```
download_and_transform_base_data.sh
```

### Isolated function data
The other part of the data includes isolated function splits, e.g a train
and test file that only includes samples for the "repeat" operation. 

To generate all the function splits, run:

```
generate_function_splits.sh
```
