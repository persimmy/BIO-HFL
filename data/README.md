# Data directory

This repository does not redistribute datasets.

Expected layouts:

- CIFAR-10: downloaded automatically by torchvision into `data/cifar` when the CIFAR scripts are run.
- DAGM2007: place the dataset under `data/dagm2007`, or pass a different path with `--dagm_data_dir`.

DAGM2007 expected layout:

```text
data/dagm2007/
  Class1/
    Train/
    Test/
  ...
  Class10/
    Train/
    Test/
```
