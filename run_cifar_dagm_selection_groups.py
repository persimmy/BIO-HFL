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


CIFAR_RESULT = ROOT / "results" / "cifar10_iid_dgcct03_selection_adverse_profile"
DAGM_RESULT = ROOT / "results" / "dagm2007_iid_dgcct01_selection_adverse_profile_lr001_localep1_edge1_epochs70"
ALGORITHMS = [
    ("dmab", "DMAB q=0.7", "dmab_q07_seed9"),
    ("greedy", "Greedy", "greedy_seed9"),
    ("gossip", "Gossip", "gossip_seed9"),
    ("oort_loss", "Oort-loss", "oort_loss_seed9"),
]


def cifar_cmd(result_dir, algorithm):
    return [
        sys.executable, str(ROOT / "main_fed.py"),
        "--snn", "--dataset", "CIFAR10", "--num_classes", "10", "--model", "VGG9",
        "--optimizer", "SGD", "--bs", "32", "--local_bs", "32", "--lr", "0.01",
        "--epochs", "50", "--local_ep", "2", "--edge_rounds", "2", "--eval_every", "1",
        "--num_users", "10", "--num_edges", "2", "--mab_k", "2", "--mab_q", "0.7",
        "--iid", "--gpu", "0", "--timesteps", "20", "--seed", "9", "--client_selection", algorithm,
        "--client_distance_profile", "extreme", "--client_distance_seed", ADVERSE_DISTANCE_SEED,
        "--client_distance_min", "30", "--client_distance_max", "800",
        "--dgc_enable", "--dgc_ratio", "0.3", "--dgc_warmup", "5", "--dgc_fp16",
        "--result_dir", str(result_dir),
    ]


def dagm_cmd(result_dir, algorithm):
    return [
        sys.executable, str(ROOT / "main_fed.py"),
        "--snn", "--dataset", "DAGM2007", "--dagm_data_dir", DAGM_DATA_DIR,
        "--num_classes", "10", "--model", "VGG9", "--optimizer", "SGD", "--bs", "64",
        "--local_bs", "32", "--lr", "0.001", "--weight_decay", "0.0005",
        "--epochs", "70", "--local_ep", "1", "--edge_rounds", "1", "--eval_every", "1",
        "--num_users", "10", "--num_edges", "2", "--mab_k", "2", "--mab_q", "0.7",
        "--iid", "--gpu", "0", "--timesteps", "20", "--seed", "9", "--client_selection", algorithm,
        "--client_distance_profile", "extreme", "--client_distance_seed", ADVERSE_DISTANCE_SEED,
        "--client_distance_min", "30", "--client_distance_max", "800",
        "--dgc_enable", "--dgc_ratio", "0.1", "--dgc_warmup", "5", "--dgc_fp16",
        "--result_dir", str(result_dir),
    ]


def print_plan():
    print("Client-selection comparison")
    print("CIFAR10: IID, DGC-CT0.3, local_ep=2, edge_rounds=2, epochs=50")
    print("DAGM2007: IID, DGC-CT0.1, lr=0.001, local_ep=1, edge_rounds=1, epochs=70")
    print("Algorithms: DMAB q=0.7, Greedy, Gossip, Oort-loss")


def main():
    print_plan()
    for algorithm, label, folder in ALGORITHMS:
        result_dir = CIFAR_RESULT / folder
        run_stage(cifar_cmd(result_dir, algorithm), "CIFAR10 {}".format(label), result_dir)
    for algorithm, label, folder in ALGORITHMS:
        result_dir = DAGM_RESULT / folder
        run_stage(dagm_cmd(result_dir, algorithm), "DAGM2007 {}".format(label), result_dir)


if __name__ == "__main__":
    main()

