"""
Generates timings for Fig.5 - environment step + RL iteration.
The script is essentially a driver for ../benchmark_training.py
"""
import os
from pathlib import Path


ROOT = Path(__file__).parent.parent.parent.parent
PATH_PROCESSED = ROOT / "data" / "processed" / "fig5"


if __name__ == "__main__":

    PATH_PROCESSED.mkdir(parents=True, exist_ok=True)

    os.execl(
        "python",
        "../benchmark_training.py",
        "--output", str(PATH_PROCESSED / "benchmark-RL-training-fast-cuda.json"),
        "--agent",
        "--env",
        "--fast_obs",
        "--fst_ents",
        "--num_envs", "32",
        "--num_tests", "10",
        "--env_device", "cuda",
        "--agent_device", "cuda"
    )
