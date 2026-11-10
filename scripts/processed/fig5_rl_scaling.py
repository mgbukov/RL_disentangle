"""
Generates timings for Fig.5 - environment step + RL iteration.
The script is essentially a driver for ../benchmark_training.py
"""
import sys
import subprocess
from pathlib import Path

import torch



ROOT = Path(__file__).parent.parent.parent.parent
PATH_PROCESSED = ROOT / "data" / "processed" / "fig5"


if __name__ == "__main__":

    PATH_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Selected device:", device)

    # Construct the command array
    cmd = [
        sys.executable,  # Ensures the same Python environment is used
        "../benchmark_training.py",
        "--output", str(PATH_PROCESSED / f"_benchmark-RL-training-fast-{device}.json"),
        "--agent",
        "--env",
        "--fast_obs",
        "--fast_ents",
        "--num_envs", "32",
        "--num_tests", "10",
        "--env_device", device,
        "--agent_device", device
    ]

    # Execute the process and wait for it to complete
    result = subprocess.run(cmd, check=True)
