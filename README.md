# BIO-HFL Codes

This repository contains the public SNN implementation for BIO-HFL experiments.
It includes the hierarchical federated learning training entrypoint, DMAB client
selection, fallback DGC / DGC-CT communication compression, energy accounting,
and reproducible scripts for the experiments reported in the paper.

This public release contains only the SNN experiments used for the reported BIO-HFL results.

## Main Components

- `main_fed.py`: SNN-only federated training entrypoint for CIFAR10 and DAGM2007.
- `models/vgg_spiking_bntt.py`: VGG9-style SNN with BNTT.
- `models/dgc_integration.py`: fallback Top-K DGC with per-client residual memory
  and optional critical-tensor transmission for DGC-CT.
- `client_selection/`: DMAB, Greedy, Gossip, and Oort-loss client selection.
- `models/initialenergy.py`: SNN computation energy and wireless communication
  energy accounting.
- `utils/`: argument parsing, CIFAR partitioning, DAGM2007 loading, and fairness metrics.

## DGC-CT Implementation Note

The validated implementation is the fallback Top-K + error-feedback path in
`models/dgc_integration.py`. DGC-CT sends selected critical tensors densely and
uses ordinary DGC for the remaining trainable parameters. The Horovod-based path
from the historical project is not included in this release and was not used for
the reported experiments.

By default, `--dgc_enable` turns on DGC-CT. Add `--dgc_disable_ct` to run ordinary
DGC without critical-tensor transmission.

## Data

CIFAR10 is downloaded by torchvision. DAGM2007 is not redistributed here. Place
DAGM2007 under:

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

You can also pass a custom DAGM2007 path with `--dagm_data_dir`.

## Environment

The experiments require PyTorch with CUDA for practical runtime. A Conda
environment template is provided in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate bio-hfl
```

The template uses `pytorch-cuda=11.8`. If your GPU driver requires a different
CUDA runtime, install the matching PyTorch build from the official PyTorch
instructions, then install the remaining packages with:

```bash
python -m pip install -r requirements.txt
```

For CPU-only smoke checks, create a normal Python environment and install the CPU
PyTorch wheels instead:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Full experiments are expected to be slow on CPU. Use `--gpu -1` only for quick
sanity checks.


## Reproducibility Notes

All public scripts set `--seed 9` and, where communication heterogeneity is used,
`--client_distance_seed 199875`. These seeds fix the Python, NumPy, and PyTorch
RNG states used by the main training process, the client-distance profile, the
initial edge/client partition, and the deterministic parts of DMAB and Greedy.
DAGM IID and Dirichlet non-IID partitions are also generated from the configured
seed.

There are several known limits to exact cross-process replay:

- Gossip uses `id(self)` as part of its local random seed in `models/Fed.py`.
  `id(self)` is a Python object address and can change from one process launch to
  another, so Gossip may select different clients even when `--seed` is fixed.
  This affects the exact client trace of Gossip baselines, but it does not affect
  the overall paper conclusion, which compares aggregate behavior across the
  same framework and energy-accounting protocol.
- CUDA/cuDNN is configured for runtime performance (`cudnn.benchmark=True` and
  `cudnn.deterministic=False`). GPU kernels can therefore introduce small
  numerical differences across hardware, drivers, PyTorch versions, or repeated
  launches.
- The default public scripts run with `parallel_workers=1` and `num_workers=0`.
  Increasing `--parallel_workers` or setting `NUM_WORKERS>0` can introduce
  additional thread scheduling or DataLoader worker-order variation.
- Oort-loss uses observed local training losses as selection scores. If training
  losses differ slightly because of GPU numerical nondeterminism, later Oort
  selections can also diverge.
- Existing `summary.csv` files cause a script stage to be skipped. Delete the
  corresponding result directory before rerunning an experiment from scratch.

For strict reruns, keep the default worker settings, use the same GPU/software
environment, and clear the target result directory before launching a script.
The nondeterministic factors listed above may cause minor client-trace or metric-level
numerical differences, but they do not change the overall conclusions of the
paper.

## Public Experiment Scripts

The public scripts are intentionally limited to the paper experiments:

| Script | Experiments |
| --- | --- |
| `run_cifar_dagm_dgc_groups.py` | CIFAR10 and DAGM2007 DGC ratio groups |
| `run_cifar_dagm_selection_groups.py` | CIFAR10 and DAGM2007 client-selection comparisons |
| `run_cifar_dagm_conventional_vs_biohfl.py` | Conventional framework vs BIO-HFL on CIFAR10 and DAGM2007 |
| `run_dagm_noniid_q_sweep.py` | DAGM2007 non-IID no-DGC DMAB q sweep |
| `run_dagm_iid_timestep_sweep.py` | DAGM2007 IID timestep sweep |
| `run_dagm_iid_warmup_ablation.py` | DAGM2007 IID DGC-CT warmup ablation |
| `run_cifar_noniid_clients40_edge_count.py` | CIFAR10 non-IID 40-client edge-server count study |

Run a script from the repository root:

```bash
python run_cifar_dagm_dgc_groups.py
```

Each script skips a stage when `summary.csv` already exists in that stage's
result directory.

## Metrics

Training writes per-round metrics to `selection_metrics.csv` and the compact
curve file `try.csv`. The logged metrics include test accuracy, macro-average F1,
selection fairness by Jain's index, total energy, SNN computation energy,
communication bytes, selected clients, per-client training losses, and elapsed
training time.


