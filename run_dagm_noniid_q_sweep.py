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


BASE_RESULT = ROOT / "results" / "dagm2007_noniid_alpha05_nodgc_adverse_profile_dmab_q_sweep_lr001_localep1_edge1_epochs70"
Q_VALUES = ["0.9", "0.7", "0.5", "0.3", "0.1"]


def q_tag(q):
    return "q" + q.replace(".", "p")


def command_for(q, result_dir):
    return [
        sys.executable, str(ROOT / "main_fed.py"),
        "--snn", "--dataset", "DAGM2007", "--dagm_data_dir", DAGM_DATA_DIR,
        "--dirichlet_alpha", "0.5", "--num_classes", "10", "--model", "VGG9",
        "--optimizer", "SGD", "--bs", "64", "--local_bs", "32", "--lr", "0.001",
        "--weight_decay", "0.0005", "--epochs", "70", "--local_ep", "1", "--edge_rounds", "1",
        "--eval_every", "1", "--num_users", "10", "--num_edges", "2", "--mab_k", "2",
        "--mab_q", str(q), "--gpu", "0", "--timesteps", "20", "--seed", "9",
        "--client_selection", "dmab", "--client_distance_profile", "extreme",
        "--client_distance_seed", ADVERSE_DISTANCE_SEED, "--client_distance_min", "30",
        "--client_distance_max", "800", "--result_dir", str(result_dir),
    ]


def main():
    print("DAGM2007 non-IID alpha=0.5 no DGC DMAB q sweep")
    for q in Q_VALUES:
        result_dir = BASE_RESULT / ("dmab_" + q_tag(q) + "_nodgc_seed9")
        run_stage(command_for(q, result_dir), "DAGM2007 non-IID alpha=0.5 no DGC DMAB q={}".format(q), result_dir)


if __name__ == "__main__":
    main()

