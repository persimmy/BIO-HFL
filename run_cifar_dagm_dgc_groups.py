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


CIFAR_RESULT = ROOT / "results" / "cifar10_iid_dgc_ratio_group_adverse_profile_dmab_q07"
DAGM_RESULT = ROOT / "results" / "dagm2007_iid_dgc_ratio_group_adverse_profile_dmab_q07_lr001_localep1_edge1_epochs70"

CIFAR_EXPERIMENTS = [
    {"name": "CIFAR10 no DGC", "folder": "nodgc_seed9", "ratio": None, "ct": False},
    {"name": "CIFAR10 DGC0.01 CT off", "folder": "dgc0p01_ct_off_seed9", "ratio": "0.01", "ct": False},
    {"name": "CIFAR10 DGC0.1 CT off", "folder": "dgc0p1_ct_off_seed9", "ratio": "0.1", "ct": False},
    {"name": "CIFAR10 DGC0.3 CT off", "folder": "dgc0p3_ct_off_seed9", "ratio": "0.3", "ct": False},
    {"name": "CIFAR10 DGC-CT0.1", "folder": "dgcct0p1_seed9", "ratio": "0.1", "ct": True},
]

DAGM_EXPERIMENTS = [
    {"name": "DAGM2007 no DGC", "folder": "nodgc_seed9", "ratio": None, "ct": False},
    {"name": "DAGM2007 DGC0.1 CT off", "folder": "dgc0p1_ct_off_seed9", "ratio": "0.1", "ct": False},
    {"name": "DAGM2007 DGC0.3 CT off", "folder": "dgc0p3_ct_off_seed9", "ratio": "0.3", "ct": False},
    {"name": "DAGM2007 DGC0.5 CT off", "folder": "dgc0p5_ct_off_seed9", "ratio": "0.5", "ct": False},
    {"name": "DAGM2007 DGC-CT0.1", "folder": "dgcct0p1_seed9", "ratio": "0.1", "ct": True},
]


def cifar_base_cmd(result_dir):
    return [
        sys.executable, str(ROOT / "main_fed.py"),
        "--snn", "--dataset", "CIFAR10", "--num_classes", "10", "--model", "VGG9",
        "--optimizer", "SGD", "--bs", "32", "--local_bs", "32", "--lr", "0.01",
        "--epochs", "50", "--local_ep", "2", "--edge_rounds", "2", "--eval_every", "1",
        "--num_users", "10", "--num_edges", "2", "--mab_k", "2", "--mab_q", "0.7",
        "--iid", "--gpu", "0", "--timesteps", "20", "--seed", "9", "--client_selection", "dmab",
        "--client_distance_profile", "extreme", "--client_distance_seed", ADVERSE_DISTANCE_SEED,
        "--client_distance_min", "30", "--client_distance_max", "800", "--result_dir", str(result_dir),
    ]


def dagm_base_cmd(result_dir):
    return [
        sys.executable, str(ROOT / "main_fed.py"),
        "--snn", "--dataset", "DAGM2007", "--dagm_data_dir", DAGM_DATA_DIR,
        "--num_classes", "10", "--model", "VGG9", "--optimizer", "SGD", "--bs", "64",
        "--local_bs", "32", "--lr", "0.001", "--weight_decay", "0.0005",
        "--epochs", "70", "--local_ep", "1", "--edge_rounds", "1", "--eval_every", "1",
        "--num_users", "10", "--num_edges", "2", "--mab_k", "2", "--mab_q", "0.7",
        "--iid", "--gpu", "0", "--timesteps", "20", "--seed", "9", "--client_selection", "dmab",
        "--client_distance_profile", "extreme", "--client_distance_seed", ADVERSE_DISTANCE_SEED,
        "--client_distance_min", "30", "--client_distance_max", "800", "--result_dir", str(result_dir),
    ]


def with_dgc(cmd, exp):
    if exp["ratio"] is None:
        return cmd
    cmd = cmd + ["--dgc_enable", "--dgc_ratio", exp["ratio"], "--dgc_warmup", "5", "--dgc_fp16"]
    if not exp["ct"]:
        cmd.append("--dgc_disable_ct")
    return cmd


def print_plan():
    print("DGC ratio experiment group")
    print("CIFAR10: IID, DMAB q=0.7, local_ep=2, edge_rounds=2, lr=0.01, T=20, epochs=50")
    print("DAGM2007: IID, DMAB q=0.7, local_ep=1, edge_rounds=1, lr=0.001, T=20, epochs=70")
    print("Profile: extreme adverse communication profile, client_distance_seed={}".format(ADVERSE_DISTANCE_SEED))


def main():
    print_plan()
    for exp in CIFAR_EXPERIMENTS:
        result_dir = CIFAR_RESULT / exp["folder"]
        run_stage(with_dgc(cifar_base_cmd(result_dir), exp), exp["name"], result_dir)
    for exp in DAGM_EXPERIMENTS:
        result_dir = DAGM_RESULT / exp["folder"]
        run_stage(with_dgc(dagm_base_cmd(result_dir), exp), exp["name"], result_dir)


if __name__ == "__main__":
    main()

