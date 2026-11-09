import sys
from pathlib import Path

import numpy as np

PATH_SOURCE_CODE = Path(__file__).parent.parent.parent.parent / Path("src_code")
sys.path.append(str(PATH_SOURCE_CODE))
from src.util import random_quantum_state



PATH_OUTPUT = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "fig18"


if __name__ == "__main__":

    PATH_OUTPUT.mkdir(exist_ok=True)

    for seed in (4, 240, 281):
        np.random.seed(seed)
        state = random_quantum_state(5, 1.0)
        np.save(str(PATH_OUTPUT / f"5q-state-seed={seed}.npy"), state)
