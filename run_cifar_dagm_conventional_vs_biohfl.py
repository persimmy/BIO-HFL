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


CIFAR_RESULT = ROOT / "results" / "cifar10_iid_conventional_vs_biohfl_adverse_profile"
DAGM_RESULT = ROOT / "results" / "dagm2007_iid_conventional_vs_biohfl_adverse_profile_lr001_localep1_edge1_epochs70"


def cifar_cmd(result_dir, algorithm, dgcct):
    cmd = [
        sys.executable, str(ROOT / "main_fed.py"),
        "--snn", "--dataset", "CIFAR10", "--num_classes", "10", "--model", "VGG9",
        "--optimizer", "SGD", "--bs", "32", "--local_bs", "32", "--lr", "0.01",
        "--epochs", "50", "--local_ep", "2", "--edge_rounds", "2", "--eval_every", "1",
        "--num_users", "10", "--num_edges", "2", "--mab_k", "2", "--mab_q", "0.7",
        "--iid", "--gpu", "0", "--timesteps", "20", "--seed", "9", "--client_selection", algorithm,
        "--client_distance_profile", "extreme", "--client_distance_seed", ADVERSE_DISTANCE_SEED,
        "--client_distance_min", "30", "--client_distance_max", "800", "--result_dir", str(result_dir),
    ]
    if dgcct:
        cmd += ["--dgc_enable", "--dgc_ratio", "0.1", "--dgc_warmup", "5", "--dgc_fp16"]
    return cmd


def dagm_cmd(result_dir, algorithm, dgcct):
    cmd = [
        sys.executable, str(ROOT / "main_fed.py"),
        "--snn", "--dataset", "DAGM2007", "--dagm_data_dir", DAGM_DATA_DIR,
        "--num_classes", "10", "--model", "VGG9", "--optimizer", "SGD", "--bs", "64",
        "--local_bs", "32", "--lr", "0.001", "--weight_decay", "0.0005",
        "--epochs", "70", "--local_ep", "1", "--edge_rounds", "1", "--eval_every", "1",
        "--num_users", "10", "--num_edges", "2", "--mab_k", "2", "--mab_q", "0.7",
        "--iid", "--gpu", "0", "--timesteps", "20", "--seed", "9", "--client_selection", algorithm,
        "--client_distance_profile", "extreme", "--client_distance_seed", ADVERSE_DISTANCE_SEED,
        "--client_distance_min", "30", "--client_distance_max", "800", "--result_dir", str(result_dir),
    ]
    if dgcct:
        cmd += ["--dgc_enable", "--dgc_ratio", "0.1", "--dgc_warmup", "5", "--dgc_fp16"]
    return cmd


def main():
    print("Conventional framework vs BIO-HFL")
    print("Conventional framework: no DGC + Gossip")
    print("BIO-HFL: DGC-CT0.1 + DMAB q=0.7")
    stages = [
        ("CIFAR10 conventional framework", cifar_cmd(CIFAR_RESULT / "conventional_nodgc_gossip_seed9", "gossip", False), CIFAR_RESULT / "conventional_nodgc_gossip_seed9"),
        ("CIFAR10 BIO-HFL", cifar_cmd(CIFAR_RESULT / "biohfl_dgcct0p1_dmab_q07_seed9", "dmab", True), CIFAR_RESULT / "biohfl_dgcct0p1_dmab_q07_seed9"),
        ("DAGM2007 conventional framework", dagm_cmd(DAGM_RESULT / "conventional_nodgc_gossip_seed9", "gossip", False), DAGM_RESULT / "conventional_nodgc_gossip_seed9"),
        ("DAGM2007 BIO-HFL", dagm_cmd(DAGM_RESULT / "biohfl_dgcct0p1_dmab_q07_seed9", "dmab", True), DAGM_RESULT / "biohfl_dgcct0p1_dmab_q07_seed9"),
    ]
    for title, cmd, result_dir in stages:
        run_stage(cmd, title, result_dir)


if __name__ == "__main__":
    main()

