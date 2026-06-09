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


BASE_RESULT = ROOT / "results" / "dagm2007_iid_dgcct01_adverse_profile_dmab_q07_timestep_sweep_lr001_localep1_edge1_epochs70"
TIMESTEPS = ["5", "10", "15", "20", "25", "30"]


def command_for(timesteps, result_dir):
    return [
        sys.executable, str(ROOT / "main_fed.py"),
        "--snn", "--dataset", "DAGM2007", "--dagm_data_dir", DAGM_DATA_DIR,
        "--iid", "--num_classes", "10", "--model", "VGG9", "--optimizer", "SGD",
        "--bs", "64", "--local_bs", "32", "--lr", "0.001", "--weight_decay", "0.0005",
        "--epochs", "70", "--local_ep", "1", "--edge_rounds", "1", "--eval_every", "1",
        "--num_users", "10", "--num_edges", "2", "--mab_k", "2", "--mab_q", "0.7",
        "--gpu", "0", "--timesteps", str(timesteps), "--seed", "9", "--client_selection", "dmab",
        "--client_distance_profile", "extreme", "--client_distance_seed", ADVERSE_DISTANCE_SEED,
        "--client_distance_min", "30", "--client_distance_max", "800",
        "--dgc_enable", "--dgc_ratio", "0.1", "--dgc_warmup", "5", "--dgc_fp16",
        "--result_dir", str(result_dir),
    ]


def main():
    print("DAGM2007 IID DGC-CT0.1 DMAB q=0.7 timestep sweep")
    for t in TIMESTEPS:
        result_dir = BASE_RESULT / ("dmab_q0p7_dgcct0p1_T{}_seed9".format(t))
        run_stage(command_for(t, result_dir), "DAGM2007 IID T={}".format(t), result_dir)


if __name__ == "__main__":
    main()

