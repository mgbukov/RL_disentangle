import os
import sys
import time
sys.path.append("..")
from tqdm import tqdm

import torch
import torch.nn.functional as F

from src.agents.pg_agent import PGAgent
from src.envs.rdm_environment import QubitsEnvironment
from src.infrastructure.logging import logText, logPlot
from src.infrastructure.util_funcs import fix_random_seeds, set_printoptions, plt_style_use
from src.policies.fcnn_policy import FCNNPolicy


def rollout(agent, steps):
    states = torch.zeros(size=(steps+1, *agent.env.obs_shape), dtype=torch.float32)
    actions = torch.zeros(size=(steps, b), dtype=torch.int64)
    rewards = torch.zeros(size=(steps, b), dtype=torch.float32)
    done = torch.zeros(size=(steps, b), dtype=torch.bool)

    # Perform parallel rollout along all trajectories.
    for i in range(steps):
        states[i] = torch.from_numpy(agent.env.observations)
        acts = agent.policy.get_action(states[i])
        actions[i] = acts
        s, r, d = agent.env.step(acts.cpu().numpy())
        rewards[i] = torch.from_numpy(r)
        done[i] = torch.from_numpy(d)

    # Permute `step` and `batch_size` dimensions
    states = states.permute(1, 0, 2)
    done = done.permute(1, 0)
    actions = actions.permute(1, 0)
    rewards = rewards.permute(1, 0)

    # Mask out the rewards after a trajectory is done.
    m = torch.roll(done, shifts=1, dims=1)
    m[:, 0] = False
    mask = ~m
    rewards = mask * rewards

    return (states, actions, rewards, mask)


# Fix the random seeds for NumPy and PyTorch, and set print options.
fix_random_seeds(seed=0)
set_printoptions(precision=5, sci_mode=False)
plt_style_use()

q = 5
steps = 40
num_test = 10
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Create file to log output during training.
log_dir = os.path.join("..", "logs", f"{q}qubits", f"timing_training_{device}")
os.makedirs(log_dir, exist_ok=True)
logfile = os.path.join(log_dir, "splitting_the_pass_to_batches.log")

# Initialize the environment.
# We always generate 1024 trajects.
env = QubitsEnvironment(num_qubits=q, epsi=1e-3, batch_size=1024)

# Initialize the policy.
input_size = env.obs_shape[1]
hidden_dims = [4096, 4096, 512]
output_size = env.num_actions
policy = FCNNPolicy(input_size, hidden_dims, output_size)
agent = PGAgent(env, policy)
optimizer = torch.optim.Adam(agent.policy.parameters(), lr=1e-4)




batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]#, 2048, 4096]
rollout_times, forward_times, backward_times = [], [], []
for i in tqdm(range(num_test)):
    bench_rollout, bench_forward, bench_backward = 0., 0., 0.

    # Set the initial state and perform policy rollout.
    agent.env.set_random_states()
    states, actions, rewards, mask = rollout(agent, steps)

    states_cuda = states.to(device)
    actions_cuda = actions.to(device)
    rewards_cuda = rewards.to(device)
    mask_cuda = mask.to(device)

    #
    start_event.record()

    logits = agent.policy(states)
    episode_entropy = agent.entropy_term(logits, actions, mask)
    q_values = agent.reward_to_go(rewards) - 0.5 * 0.01 * episode_entropy
    q_values -= agent.reward_baseline(q_values, mask)
    nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
    weighted_nll = torch.mul(mask * nll, q_values)
    loss = torch.sum(weighted_nll) / torch.sum(mask)

    end_event.record()



    tic = time.time()
    for b in tqdm(batch_sizes):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Set the initial state and perform policy rollout.
        agent.env.set_random_states()

