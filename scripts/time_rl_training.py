# Measure how long it takes the agent to perform one iteration of training.
# Training iterations consists of three major computations:
#   * policy rollout to collect samples
#   * forward pass to compute the loss
#   * backward pass to update the weights
#
# Measure each of the components by running multiple iteration steps and then
# averaging the total time.
#
# Perform the measurement across different values for the batch size in order
# to observe whether using different batch sizes would have a better effect.


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

# Fix the random seeds for NumPy and PyTorch, and set print options.
fix_random_seeds(seed=0)
set_printoptions(precision=5, sci_mode=False)
plt_style_use()

q = 5
num_test = 100
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Create file to log output during training.
log_dir = os.path.join("..", "logs", f"{q}qubits", f"timing_training_{device}")
os.makedirs(log_dir, exist_ok=True)
logfile = os.path.join(log_dir, "testing.log")


batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
rollout_times, forward_times, backward_times = [], [], []
for b in tqdm(batch_sizes):
    # Create the environment.
    env = QubitsEnvironment(num_qubits=q, epsi=1e-3, batch_size=b)

    # Initialize the policy.
    input_size = 2 ** (q + 1)
    hidden_dims = [4096, 4096, 512]
    output_size = env.num_actions
    policy = FCNNPolicy(input_size, hidden_dims, output_size)
    agent = PGAgent(env, policy)
    optimizer = torch.optim.Adam(agent.policy.parameters(), lr=1e-4)

    bench_rollout, bench_forward, bench_backward = 0., 0., 0.
    tic = time.time()
    for i in tqdm(range(num_test)):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Set the initial state and perform policy rollout.
        agent.env.set_random_states()

        start_event.record()
        states, actions, rewards, done = agent.rollout(steps=40)
        end_event.record()
        torch.cuda.synchronize()
        bench_rollout += start_event.elapsed_time(end_event)

        mask = agent.generate_mask(done)
        states = states[:, :-1, :]

        # Compute the loss.
        start_event.record()
        logits = agent.policy(states)
        end_event.record()
        torch.cuda.synchronize()
        bench_forward += start_event.elapsed_time(end_event)

        episode_entropy = agent.entropy_term(logits, actions, mask)
        q_values = agent.reward_to_go(rewards) - 0.5 * 0.01 * episode_entropy
        q_values -= agent.reward_baseline(q_values, mask)
        nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
        weighted_nll = torch.mul(mask * nll, q_values)
        loss = torch.sum(weighted_nll) / torch.sum(mask)

        # Perform backward pass.
        optimizer.zero_grad()

        start_event.record()
        loss.backward()
        end_event.record()
        torch.cuda.synchronize()
        bench_backward += start_event.elapsed_time(end_event)

        total_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in agent.policy.parameters()]))
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 10.)
        optimizer.step()

    toc = time.time()
    logText(f"batch_size = {b}", logfile)
    logText(f"  Rollout takes on average: {bench_rollout/num_test/1000: .3f} sec", logfile)
    logText(f"  Forward pass takes on average: {bench_forward/num_test/1000: .3f} sec", logfile)
    logText(f"  Backward pass takes on average: {bench_backward/num_test/1000: .3f} sec", logfile)
    logText(f"  One iteration takes: {(toc-tic)/num_test: .3f} sec", logfile)

    rollout_times.append(bench_rollout / num_test / 1000)
    forward_times.append(bench_forward / num_test / 1000)
    backward_times.append(bench_backward / num_test / 1000)


logPlot(
    figname=os.path.join(log_dir, "time_vs_batch_size"),
    xs=[batch_sizes, batch_sizes, batch_sizes],
    funcs=[rollout_times, forward_times, backward_times],
    legends=["rollout", "forward", "backward"],
    labels={"x":"Batch Size", "y":"Seconds"},
    lw=[3., 3., 3.],
    figtitle="Policy Training Loss",
)
