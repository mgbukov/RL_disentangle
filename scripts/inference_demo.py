"""Script that shows a sample usage of the RL agents (including the 12 and 16q ones)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))
from src.qenv import QEnv
from src.stategen import sample_haar_full, sample_haar_generalized
from src.networks import TransformerPE_2qRDM



if __name__ == "__main__":

    L = int(sys.argv[1])
    assert L in (4, 5, 6, 12, 16)

    # Generate a random state:
    if L in (4, 5, 6):
        state = sample_haar_full(L)
    else:
        state = sample_haar_generalized(L, 2, 4, 4.1, 4.1)

    # Instantiate an RL environment.
    env = QEnv(L, 1, obs_fn='rdm2m')
    env.reset()
    env.set_states(state.reshape((1,) + (2,) * L ))

    # Load the policy network
    with open(ROOT / "models" / f"{L}q-policy-config.json", mode='rt') as f:
        policy_config = json.load(f)
    policy = TransformerPE_2qRDM(
        in_dim = env.single_observation_space.shape[1],
        embed_dim=policy_config["embed_dim"],
        dim_mlp=policy_config["dim_mlp"],
        n_heads=policy_config["attn_heads"],
        n_layers=policy_config["transformer_layers"]
    )
    state_dict = torch.load(ROOT / "models" / f"{L}q-policy-statedict.pt",
                            map_location="cpu", weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()

    # Do a rollout
    trajectory = []
    success = False
    with torch.no_grad():
        for _ in range(100):
            observation = env.obs_fn(env.simulator.states)
            logits = policy(observation)[0]
            probs = Categorical(logits=logits).probs
            a = np.argmax(probs.numpy())
            trajectory.append(env.actions[a])
            o, r, t, tr, i = env.step([a], reset=False)
            if torch.all(t):
                success = True
                break

    # The selected actions are in `trajectory`
    print(trajectory)