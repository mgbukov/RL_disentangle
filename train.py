import numpy as np
import matplotlib.pyplot as plt
import torch
from environment_main import QubitsEnvironment
from approximators import PolicyNetwork, ValueNetwork

torch.manual_seed(7)
np.random.seed(7)

def generate_samples(model, N=30, L=20, env=None):
    choose = torch.multinomial
    if env is None:
        env = QubitsEnvironment(2)
        env.set_random_state()
    A = np.arange(len(env.state))

    s0 = env.state.copy()
    states, rewards, ends = [], [], []
    for _ in range(N):
        env.state = s0
        states.append(s0)
        for k in range(L):
            print(k)
            allowed_actions = torch.tensor(env.allowed_actions())
            S = torch.tensor(env.state.view(np.float32))
            policy = model(S, allowed_actions)
            action = choose(policy, 1)
            nextstate, reward, done = env.step(action)
            states.append(nextstate)
            rewards.append(reward)
            ends.append(done)
            if done:
                states.append([nextstate] * (L - k + 1))
                rewards.append([0] * (L - k + 1))
                ends.append([done] * (L - k + 1))
                break
    states = np.array(states).reshape((N, L, -1))
    rewards = np.array(rewards).reshape((N, L))
    ends = np.array(ends).reshape((N, L))
    return states, rewards, ends


e = QubitsEnvironment()
p = PolicyNetwork(8, 9)
samples = generate_samples(p)
print(samples)
