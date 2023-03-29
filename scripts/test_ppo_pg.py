import sys
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fix_random_seeds(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)

def set_printoptions(precision, sci_mode):
    torch.set_printoptions(precision=precision, sci_mode=sci_mode)
    np.set_printoptions(precision=precision, suppress=~sci_mode)

q = 5
seed = 0
device = torch.device("cpu")
dtype = torch.float64

fix_random_seeds(seed)
set_printoptions(precision=14, sci_mode=False)


# # Load the neural net params.
# from src.policies.fcnn_policy import FCNNPolicy
# net = FCNNPolicy.load("policy_5000.bin") # entropy = 0.46
# net.to(device)

# Initialize a random neural net and an optimizer.
net = nn.Sequential(
    nn.Linear(2**(q+1), 256), # input_size = 2**(q+1) = 64
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, q*(q-1)), # num_actions = q*(q-1) = 20
)

set_printoptions(precision=14, sci_mode=False)

dtype = torch.float64
net.to(device, dtype=dtype)
optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)


# Load the data.
states = torch.load("states.torch").to(device, dtype=dtype)     # the states visited by the agent
actions = torch.load("actions.torch").to(device)   # the actions taken by the agent
q_values = torch.load("q_values.torch").to(device, dtype=dtype) # the returns obtained for every taken action

# states, actions, and q_values have shape (batch_size, steps)
print("actions shape:", actions.shape) # (1024, 40)

# Q values equal 0. when the episode ends, i.e. they are masked.
mask = ~(q_values == 0)
print("q_values[-3]:\n", q_values[-3])
print("mask[-3]:\n", mask[-3])


def compare_ppo_to_pg(q_values):
    # Computing the PG objective and backward pass.
    logits = net(states) # the un-normalized logits for the action set at every time-step.
    nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none") # negative log-likelihood, i.e. -logp
    weighted_nll = torch.mul(nll, q_values)
    loss_pg = torch.sum(weighted_nll) / torch.sum(mask)
    optimizer.zero_grad()
    loss_pg.backward()
    total_norm_pg = torch.norm(
        torch.stack([torch.norm(p.grad) for p in net.parameters()]))

    print(f"  PG loss value: {loss_pg.item():.8f}")
    print(f"  PG total grad norm: {total_norm_pg.item():.8f}")

    # Computing the PPO objective and backward pass.
    with torch.no_grad(): # compute the old logP without building a backwards graph
        logits = net(states)
        logp_old = -F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")

    K = 1
    for k in range(K):
        logits = net(states) # recompute the logits in order to build a backwards graph
        logp = -F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
        ro = torch.exp(logp_old - logp)
        # clip_adv = torch.clip(ro, 1-clip_epsi, 1+clip_epsi) * q_values
        # loss_ppo = torch.sum(torch.min(ro * q_values), clip_adv) / torch.sum(mask)
        loss_ppo = torch.sum(ro * q_values) / torch.sum(mask)

        optimizer.zero_grad()
        loss_ppo.backward()
        total_norm_ppo = torch.norm(
            torch.stack([torch.norm(p.grad) for p in net.parameters()]))

    print(f"  PPO loss value: {loss_ppo.item():.8f}")
    print(f"  PPO total grad norm: {total_norm_ppo.item():.8f}")

    # Compute the PPO objective using non-logarithmic scale.
    with torch.no_grad():
        logits = net(states)
        probs_old = F.softmax(logits, dim=-1)
        probs_old = probs_old.gather(index=actions.unsqueeze(dim=2), dim=2).squeeze(dim=2)

    K = 1
    for k in range(K):
        logits = net(states)
        probs = F.softmax(logits, dim=-1)
        probs = probs.gather(index=actions.unsqueeze(dim=2), dim=2).squeeze(dim=2)
        eps = torch.finfo(torch.float32).eps
        probs = torch.clip(probs, min=eps)
        ro = probs_old / probs
        loss_ppo2 = torch.sum(ro * q_values) / torch.sum(mask)

        optimizer.zero_grad()
        loss_ppo2.backward()
        total_norm_ppo2 = torch.norm(
            torch.stack([torch.norm(p.grad) for p in net.parameters()]))

    print(f"  PPOv2 loss value: {loss_ppo2.item():.8f}")
    print(f"  PPOv2 total grad norm: {total_norm_ppo2.item():.8f}")

    # Note that the loss value for PG is not the same as the loss value for PPO.
    # For pg we use logP * Q, where logP is approximately log(1/20)=-3. for a
    # random policy. For ppo we use ro * Q, where ro = P_old / P, which is
    # equal to 1. for k=1.
    # When we compute the gradient however, we obtain the same results.
    # PG gradient differentiates logP, while PPO gradient differentiates 1/P.
    print(f"  loss_pg / loss_ppo: {loss_pg / loss_ppo}")
    print(f"  grad_norm_pg / grad_norm_ppo: {total_norm_pg / total_norm_ppo}\n")


# Using raw Q values.
print("\nCompare PPO and PG using raw Q values")
compare_ppo_to_pg(q_values)


# Using Q values with baseline.
def calc_baseline(returns, mask, device):
    batch_size, _ = returns.shape
    baseline = torch.sum(mask * returns, dim=0, keepdim=True) / torch.maximum(
        torch.sum(mask, dim=0), torch.Tensor([1]).to(device))
    baseline = mask * torch.tile(baseline, dims=(batch_size, 1))
    return baseline

baseline = calc_baseline(q_values, mask, device)
print("Compare PPO and PG using Q values with baseline")
compare_ppo_to_pg(q_values - baseline)

# Some problems arise when we baseline the Q values.
# Summing the values in different ways produces different results.
# Note that we are not using a state-dependent baseline, e.g. V function, but
# rather we are just subtracting the mean. This means that the mean of the tensor
# is now 0. Summing the values results in a small number that is close to 0, so
# we are probably having some issues with machine precision.
print("Summing the numbers in the `adv` tensor in several different ways:")
adv = q_values - baseline
print(torch.sum(torch.sum(adv, dim=1))) # sum along x-axis and then along y-axis
print(torch.sum(torch.sum(adv, dim=0))) # sum along y-axis and then along x-axis
print(torch.sum(adv)) # default sum
print(sum(adv.ravel())) # using the built-in sum returns a wildly different result
adv_np = adv.cpu().numpy()
print(np.sum(np.sum(adv_np, axis=1))) # same with numpy..
print(np.sum(np.sum(adv_np, axis=0)))
print(np.sum(adv_np))
print(sum(adv_np.ravel()))

