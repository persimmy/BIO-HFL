# Public Experiment List

All public scripts use the SNN VGG9 model, seed 9, and the extreme adverse
communication profile with `client_distance_seed=199875` unless noted otherwise.
DMAB is passed as `--client_selection dmab`; internally it uses the same selector
as the historical `mab` alias.

## 1. CIFAR10 and DAGM2007 DGC Ratio Groups

Script: `run_cifar_dagm_dgc_groups.py`

CIFAR10 IID, `local_ep=2`, `edge_rounds=2`, `lr=0.01`, `epochs=50`, `T=20`,
`DMAB q=0.7`:

- no DGC
- DGC0.01, CT off
- DGC0.1, CT off
- DGC0.3, CT off
- DGC-CT0.1

DAGM2007 IID, `local_ep=1`, `edge_rounds=1`, `lr=0.001`, `epochs=70`, `T=20`,
`DMAB q=0.7`:

- no DGC
- DGC0.1, CT off
- DGC0.3, CT off
- DGC0.5, CT off
- DGC-CT0.1

## 2. Client-Selection Comparisons

Script: `run_cifar_dagm_selection_groups.py`

Algorithms: DMAB q=0.7, Greedy, Gossip, and Oort-loss.

- CIFAR10 IID uses DGC-CT0.3.
- DAGM2007 IID uses DGC-CT0.1.

## 3. Conventional Framework vs BIO-HFL

Script: `run_cifar_dagm_conventional_vs_biohfl.py`

For both CIFAR10 and DAGM2007:

- Conventional framework: no DGC + Gossip
- BIO-HFL: DGC-CT0.1 + DMAB q=0.7

## 4. DAGM2007 Non-IID q Sweep

Script: `run_dagm_noniid_q_sweep.py`

DAGM2007 Dirichlet non-IID `alpha=0.5`, no DGC, `lr=0.001`, `local_ep=1`,
`edge_rounds=1`, `epochs=70`, `T=20`:

- DMAB q=0.9
- DMAB q=0.7
- DMAB q=0.5
- DMAB q=0.3
- DMAB q=0.1

## 5. DAGM2007 IID Timestep Sweep

Script: `run_dagm_iid_timestep_sweep.py`

DAGM2007 IID, DGC-CT0.1, DMAB q=0.7, `lr=0.001`, `local_ep=1`,
`edge_rounds=1`, `epochs=70`:

- T=5
- T=10
- T=15
- T=20
- T=25
- T=30

## 6. DAGM2007 IID Warmup Ablation

Script: `run_dagm_iid_warmup_ablation.py`

DAGM2007 IID, DGC-CT0.1, DMAB q=0.7, `lr=0.001`, `local_ep=1`,
`edge_rounds=1`, `epochs=70`, `T=20`:

- warmup=0
- warmup=5
- warmup=10
- warmup=15

## 7. CIFAR10 Non-IID 40-Client Edge-Server Count

Script: `run_cifar_noniid_clients40_edge_count.py`

CIFAR10 Dirichlet non-IID `alpha=0.5`, DGC-CT0.3, DMAB q=0.3, `local_ep=2`,
`edge_rounds=2`, `epochs=50`, `T=20`:

- 40 clients, 2 edge servers, k=8 clients per edge
- 40 clients, 4 edge servers, k=4 clients per edge
- 40 clients, 8 edge servers, k=2 clients per edge
