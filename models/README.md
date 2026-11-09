
## Directory Structure
```
├── fig7-8/                     # Contains the 4q agent that is used to plot Fig. 7 & Fig. 8
│   └── 4q-agent.pt             # This file is different from ../4q-agent.pt !
│
├── 4q-agent.pt                 # These are files from from the first submittion of the paper
├── 5q-agent.pt                 # and are kept, because they are used to plot the figures.
├── 6q-agent.pt                 # In order to be loaded, the client must have the
│                               # source code of the project, because a custom class
│                               # is serialized (PPOAgent).
|                               # They can be loaded with `torch.load()`.
│
├── 4q-policy-config.json       # These files describe the parameters with which
├── 5q-policy-config.json       # the policy network has to be initialized in
├── 6q-policy-config.json       # order to load that PyTorch statedict files
├── 12q-policy-config.json
├── 16q-policy-config.json
│
│
├── 4q-policy-statedict.pt      # These are the trained parameters of the
├── 5q-policy-statedict.pt      # policy networks of the RL agents.
├── 6q-policy-statedict.pt
├── 12q-policy-statedict.pt
├── 16q-policy-statedict.pt
```

## Notes
The agent in `fig7-8/` directory if different from the final agent.
We keep it, because Figures 7, 8 are plotted using it. The agent is from commit
[058b01f32e1af1d2a2d5a45f03cba98f2f02c849](https://github.com/mgbukov/RL_disentangle/tree/058b01f32e1af1d2a2d5a45f03cba98f2f02c849/logs/4q_10000_iters_haar_unif2_1024envs),
experiment **4q_10000_iters_haar_unif2_1024envs**.


## Loading
Essentially you must instantiate a Policy network with the JSON configs and
then load the trained parameters from the statedict files. Example:

```python:
# Initialized only to get the number of inputs per single observation
env = QEnv(5, 1, obs_fn='rdm2m')
env.reset()

# Loading the 5q agent
with open("models/5q-policy-config.json", mode='rt') as f:
  policy_config = json.load(f)
policy = TransformerPE_2qRDM(
  in_dim=env.single_observation_space.shape[1],
  embed_dim=policy_config["embed_dim"],
  dim_mlp=policy_config["dim_mlp"],
  n_heads=policy_config["attn_heads"],
  n_layers=policy_config["transformer_layers"]
)
state_dict = torch.load("models/5q-policy-statedict.pt")
policy.load_state_dict(state_dict)
policy.eval()
```

See `src_code\scripts\inference_demo.py` for more