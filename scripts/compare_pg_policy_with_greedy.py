import sys
sys.path.append("..")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.envs.rdm_environment import QubitsEnvironment
from src.policies.fcnn_policy import FCNNPolicy
from src.infrastructure.util_funcs import plt_style_use, fix_random_seeds, set_printoptions
from src.infrastructure.logging import logPlot

plt_style_use()
fix_random_seeds(seed=0)
set_printoptions(5, False)

policy = FCNNPolicy.load("../data/5qubits/policy.bin")

q = 5
batch_size = 1024
env = QubitsEnvironment(num_qubits=q, epsi=1e-3, batch_size=batch_size)

KL = []
num_tests = 1
steps = 25
for _ in tqdm(range(num_tests)):
    env.set_random_states()

    # RL agent.
    for i in tqdm(range(steps)):
        # print(f"step {i}")
        # Policy probability.
        s = env.states.reshape(batch_size, -1)
        s = np.hstack([s.real, s.imag]).astype(np.float32)
        act = policy.get_action(torch.from_numpy(s))
        states = torch.from_numpy(s).to(policy.device)
        policy_logits = policy(states)
        policy_probs = F.softmax(policy_logits, dim=1)

        # Greedy probability.
        states = env.states.copy()
        curr_ent = env.entropy().mean(axis=1, keepdims=True)
        env.states = np.repeat(np.array(states), env.num_actions, axis=0)
        actions = np.tile(np.array(list(env.actions.keys())), (batch_size,))
        # actions = np.array(list(env.actions.keys()))
        next_states, _, _ = env.step(actions)
        costs = env.entropy().mean(axis=1)           # shape (num_act * B,)
        costs = costs.reshape(env.num_actions, -1).T # shape (B, num_act)
        delta = curr_ent - costs
        delta = delta - delta.min(axis=1, keepdims=True)
        assert np.all(delta > -1e-4) # assert that all actions reduce the entanglement
        delta = np.maximum(delta, 0)
        delta = delta / delta.max(axis=1, keepdims=True) * 100
        logits = torch.from_numpy(delta)
        greedy_probs = F.softmax(logits, dim=1)

        # Compute the KL-divergence between the policy output and the greedy policy.
        log_policy = F.log_softmax(policy_logits, dim=1)
        log_greedy = F.log_softmax(logits, dim=1)
        diff = F.kl_div(log_policy, log_greedy, log_target=True, reduction="batchmean")
        KL.append(diff.item())

        # Expand here a full beam search procedure.
        # Calculate the beam search policy.
        #

        # print("  policy:", policy_probs)
        # print("  greedy:", greedy_probs)
        # print("    logits:", logits)
        # print("  KL-divergence:", diff)

        # Advance the environment to the next state.
        env.states = states
        acts = torch.multinomial(policy_probs, 1).squeeze(dim=-1)
        _ = env.step(acts)
        if np.all(env.disentangled()):
            # print("Disentangled!")
            break



logPlot(
    figname="KL_div.png",
    funcs=[KL],
    legends=["KL divergence"],
    labels={"x":"episode step", "y":"KL-divergence"},
    lw=[3.],
)