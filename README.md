# RL disentangle


### 1. Overview
***
This repository contains the source code from the paper "Reinforcement
Learning to Disentangle Multiqubit Quantum States from Partial Observations."
and a demo of the RL Agents in a form of interactive Jupyter Notebook.

### 2. Structure
***

* **`agents/`**<br>
 This dir contains symbolic links to trained RL agents.
 Symbolic links are UNIX-like and are not compatible with Windows OS.
 If you are using Windows, delete the symbolic links after cloning the repo and
 copy all `agent.pt` files from `logs/` manually into `agents/`
 The agents are serialized instances of `src.agent.PPOAgent` class.

* **`data/`**<br>
Contains accuracy stats for the RL agents in JSON format. These stats are
used to generate the figures in the paper.

* **`logs/`**<br>
  This directory contains the text logs, various plots and checkpointed agents
  from the RL training.

* **`qiskit/`**<br>
   Contains interface code for NISQ devices

* **`scripts/`**<br>
   Contains Python & Bash scripts + Sample code

* **`src/`**<br>
    Contains the source code

* **`tests/`**<br>
    Contains tests for NISQ interface code

### 3. How to Use?
***
1. Clone the repo
2. Create new Conda (or Python) envionment & install packages from **requirements.txt** using:
`conda install --yes --file requirements.txt`
3. Check the demo script in `scripts/sample.py` and the Interative Notebook `demo.ipynb`

Essentially you must instantiate the RL environment, load the agent and then
do a rollout. The snippet bellow shows a use case for 5 qubits system:

```python:
# Instantiate an RL environment. 
num_qubits = int(np.log2(state.size))
env = QuantumEnv(num_qubits, 1, obs_fn='rdm_2q_mean_real')
env.reset()

# Set the environment's state (assuming that `state` is a NumPy
# array that holds the quantum state we want to disentangle)
shape = (2,) * num_qubits
env.simulator.states = np.expand_dims(state.reshape(shape), 0)

# Load the agent
agent = torch.load("agents/5q-agent.pt")

# Do a rollout
trajectory = []
success = False
for _ in range(100):
    observation = torch.from_numpy(env.obs_fn(env.simulator.states))
    probs = agent.policy(observation).probs[0]
    a = np.argmax(probs.cpu().numpy())
    trajectory.append(env.simulator.actions[a])
    o, r, t, tr, i = env.step([a], reset=False)
    if np.all(t):
        success = True
        break

# The selected actions are in `trajectory`
```


### 4. How to Train the Agents?
***
Check & run the script `scripts/train.sh` - it calls the Python
script `run.py` with the hyperparameters used in the paper for 4,5 and 6 qubit
agents. Our training was done on an 8-core CPU and NVidia Tesla T4 GPU.
Training times were approximately:

*  25 minutes for the 4 qubit agent
*  10 hours for the 5 qubit agent
*  60 hours for the 6 qubit agent

