import time

import numpy as np

import context
from src.quantum_env import QuantumEnv


NUM_ROLLOUTS = 10
NUM_QUBITS = 15
NUM_ENVS = 64


if __name__ == "__main__":
    base_env = QuantumEnv(NUM_QUBITS, NUM_ENVS, 1e-2, 240, obs_fn="rdm_2q_mean_real", fast_obs=False)
    fast_env = QuantumEnv(NUM_QUBITS, NUM_ENVS, 1e-2, 240, obs_fn="rdm_2q_mean_real", fast_obs=True)
    base_env.reset()
    fast_env.reset()

    initial_states = base_env.simulator.states
    fast_env.set_states(initial_states)

    A = base_env.single_action_space.n
    actions = [np.random.choice(A, size=NUM_ENVS) for _ in range(NUM_ROLLOUTS)]

    tic = time.time()
    for i in range(NUM_ROLLOUTS):
        base_env.step(actions[i])
    toc = time.time()

    elapsed = time.strftime("%H:%M:%S", time.gmtime(int(toc - tic)))
    print(f"Time for {NUM_ROLLOUTS} rollouts with `fast_obs=False`:\t", elapsed, '\n')

    tic = time.time()
    for i in range(NUM_ROLLOUTS):
        fast_env.step(actions[i])
    toc = time.time()

    elapsed = time.strftime("%H:%M:%S", time.gmtime(int(toc - tic)))
    print(f"Time for {NUM_ROLLOUTS} rollouts with `fast_obs=True`:\t", elapsed, '\n')