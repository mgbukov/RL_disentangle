from collections import defaultdict

import numpy as np

from src.envs.rdm_environment import QubitsEnvironment
from src.utils.logger import Logger
from src.utils.util_funcs import fix_random_seeds


def main():
    seed = 0
    fix_random_seeds(seed)

    # Create the environment.
    num_qubits = 4
    epsi = 1e-3
    batch_size = 10
    env = QubitsEnvironment(num_qubits, epsi=epsi, batch_size=batch_size)

    log_dir = "logs/4qubits/exhaustive"
    logger = Logger(log_dir)

    path_len = []
    num_tests = 10
    for i in range(num_tests):
        print("running test: ", i+1)
        env.set_random_state()
        res = env.compute_best_path()
        path_len.extend([len(x["paths"][0]) for x in res])

    logger.logHistogram(path_len, np.arange(max(path_len)+1), "histogram_of_pathLens.png")

    path_len_dict = defaultdict(lambda: 0)
    for val in path_len:
        path_len_dict[val] += 1

    logger.setLogTxtFilename("exhaustive_search.txt")
    logger.verboseTxtLogging(True)

    for i in range(1, max(path_len) + 1):
        logger.logTxt("number of states solvable in {:d} step(s): {:d}".format(i, path_len_dict[i]))



if __name__ == "__main__":
    main()