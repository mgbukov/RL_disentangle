import numpy as np
import os
import sys

# Find the the absolute path to project directory
dirname = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.realpath(os.path.join(dirname, os.path.pardir))
sys.path.append(project_root)
from qiskit.helpers import *
from tests.test_qiskit_helpers import *


np.set_printoptions(precision=6, suppress=True)

# Load diverging states
diverging_states = np.load(
    os.path.join(os.path.dirname(__file__), "diverging-states.npy"))

# Dictionary objects, keys = ['states', 'actions', 'entanglements',
#                             'Us', 'RDMs', 'preswaps', 'postswaps']
# IMPORTANT
# The Us between qiskit and RL env rollouts may have columns 1,2 swapped,
# if qiskit's U gate is premultiplied with swap gate on the right.
qiskit_rollout = do_qiskit_rollout(diverging_states[-1], "ordered")
rlenv_rollout = do_rlenv_rollout(diverging_states[-1], "ordered")

print("\nFidelity between qiskit and RLenv rolllouts:")
i = 0
for x, y in zip(qiskit_rollout['states'], rlenv_rollout['states']):
    print(f"states {i}:", fidelity(x, y))
    i +=1