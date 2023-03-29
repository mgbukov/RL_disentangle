import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.agents.pg_agent import PGAgent
from src.envs.rdm_environment import QubitsEnvironment
from src.infrastructure.logging import logPlot
from src.infrastructure.util_funcs import fix_random_seeds, plt_style_use
from src.policies.fcnn_policy import FCNNPolicy


fix_random_seeds(seed=0)
plt_style_use()

b = 1 # batch size
policy = FCNNPolicy.load("../data/5qubits/policy.bin")
# policy = FCNNPolicy.load("../data/5qubits/policy_stoch.bin")
env = QubitsEnvironment(num_qubits=5, epsi=1e-3, batch_size=b, stochastic=True)
agent = PGAgent(env, policy)

result = {}
stochastic_epsi = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]

steps = 400

env.set_random_states()
start_states = env.states.copy()
for epsi in tqdm(stochastic_epsi):
    env.states = start_states
    env.stochastic_eps = float(epsi)

    entropies = [] # shape (steps, b)
    done = [] # shape (steps, b)
    for step in tqdm(range(steps)):
        batch = env.states.reshape(b, -1)
        batch = np.hstack([batch.real, batch.imag]).astype(np.float32)
        batch = torch.from_numpy(batch)
        acts = policy.get_action(batch, greedy=True).cpu().numpy()
        s, r, d = env.step(acts)
        curr_entropies = env.entropy() # shape (b, L)
        entropies.append(curr_entropies.mean(axis=-1)) # shape (b,)
        done.append(d)

    entropies = np.mean(entropies, axis=-1) # shape (steps,)        # NOTE: plot std also
    std = np.std(entropies, axis=-1)
    done = np.mean(done, axis=-1)

    result[epsi] = {
        "entropies": entropies,
        "std": std,
        "done": done,
    }


logPlot(
    figname="plots/entanglement_per_step_v4.png",
    funcs=[r["entropies"] for r in result.values()],
    # fills=[(r["entropies"]-0.5*r["std"], r["entropies"]+0.5*r["std"]) for r in result.values()],
    legends=[f"stochastic_{epsi:.1e}" for epsi in stochastic_epsi],
    labels={"x":"Episode step", "y":"Average entanglement of the system"},
    lw=[3.],
    figtitle="Entanglement per step",
)

logPlot(
    figname="plots/solved_per_step_v4.png",
    funcs=[r["done"] for r in result.values()],
    legends=[f"stochastic_{epsi:.1e}" for epsi in stochastic_epsi],
    labels={"x":"Episode step", "y":"Percentage of solved systems"},
    lw=[3.],
    figtitle="Solved per step",
)


