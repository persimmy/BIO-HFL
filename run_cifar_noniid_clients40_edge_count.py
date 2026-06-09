import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ADVERSE_DISTANCE_SEED = "199875"
DAGM_DATA_DIR = "data/dagm2007"


def is_complete(result_dir):
    summary = Path(result_dir) / "summary.csv"
    return summary.exists() and summary.stat().st_size > 0


def run_stage(cmd, title, result_dir):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if is_complete(result_dir):
        print("\n=== Skipping completed stage: {} ===".format(title))
        print("Found {}".format(result_dir / "summary.csv"))
        return
    print("\n=== {} ===".format(title))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


BASE_RESULT = ROOT / "results" / "cifar10_noniid_alpha05_clients40_edge_count_dgcct03_dmab_q03"
EDGE_CONFIGS = [
    {"name": "40 clients, 2 edge servers, k=8", "folder": "clients40_edges2_k8_seed9", "num_edges": "2", "mab_k": "8"},
    {"name": "40 clients, 4 edge servers, k=4", "folder": "clients40_edges4_k4_seed9", "num_edges": "4", "mab_k": "4"},
    {"name": "40 clients, 8 edge servers, k=2", "folder": "clients40_edges8_k2_seed9", "num_edges": "8", "mab_k": "2"},
]


def command_for(edge_cfg, result_dir):
    return [
        sys.executable, str(ROOT / "main_fed.py"),
        "--snn", "--dataset", "CIFAR10", "--dirichlet_alpha", "0.5",
        "--num_classes", "10", "--model", "VGG9", "--optimizer", "SGD",
        "--bs", "32", "--local_bs", "32", "--lr", "0.01", "--epochs", "50",
        "--local_ep", "2", "--edge_rounds", "2", "--eval_every", "1",
        "--num_users", "40", "--num_edges", edge_cfg["num_edges"], "--mab_k", edge_cfg["mab_k"],
        "--mab_q", "0.3", "--gpu", "0", "--timesteps", "20", "--seed", "9",
        "--client_selection", "dmab", "--client_distance_profile", "extreme",
        "--client_distance_seed", ADVERSE_DISTANCE_SEED, "--client_distance_min", "30",
        "--client_distance_max", "800", "--dgc_enable", "--dgc_ratio", "0.3",
        "--dgc_warmup", "5", "--dgc_fp16", "--result_dir", str(result_dir),
    ]


def main():
    print("CIFAR10 non-IID alpha=0.5, 40-client edge-server count experiment")
    print("Shared config: DGC-CT0.3, DMAB q=0.3, local_ep=2, edge_rounds=2, epochs=50")
    for edge_cfg in EDGE_CONFIGS:
        result_dir = BASE_RESULT / edge_cfg["folder"]
        run_stage(command_for(edge_cfg, result_dir), edge_cfg["name"], result_dir)


if __name__ == "__main__":
    main()

