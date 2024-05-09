# RL disentangle


### 1. Overview
***
This repository contains the code for the paper "Reinforcement Learning to
Disentangle Multiqubit Quantum States from Partial Observations."

### 2. Structure
***

* **`agents/`**<br>
 This dir contains symbolic links to trained RL agents.
 Symbolic links are UNIX-like and are not compatible with Windows OS.
 If you are using Windwows, delete the symbolic links after cloning the repo and
 copy the `.pt` files from `logs/` manually into `agents/`
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


### 3. How to Run?
***
1. Clone the repo
2. Create new Conda (or Python) envionment & install packages from **requirements.txt** using:
`conda install --yes --file requirements.txt`
3. Check the demo script in `scripts/sample.py` and the Interative Notebook `demo.ipynb`


### 4. How to Train the Agents?
***
Check & run the script `scripts/train.sh`. It contains the call to the Python
script `run.py` with the hyperparameters used in the paper for 4,5 and 6 qubit
agents.
