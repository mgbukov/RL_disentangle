import sys
sys.path.append("..")

import numpy as np
import torch
import torch.nn.functional as F

from src.envs.rdm_environment import QubitsEnvironment
from src.agents.expert import SearchExpert_OLD


# log_dir = "../logs/5qubits/test/rollout_test"
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)


psi =  [[[[[ 2.3278525e-01-1.0000000e-08j,  1.4709601e-01-1.8735868e-01j],
           [ 1.2148964e-01+1.5230810e-02j,  2.2766429e-01+1.3520736e-01j]],
           [[ 1.3144670e-01+2.2850233e-01j, -1.6661417e-01+7.4815600e-02j],
           [ 1.2056164e-01+9.5626000e-03j,  8.2250000e-05-3.9850630e-02j]]],

           [[[ 4.4694680e-02-1.1587433e-01j,  1.1847973e-01-1.2775960e-01j],
           [ 1.0558201e-01-1.7202458e-01j,  4.9752460e-02+2.8246403e-01j]],
           [[ 1.0713805e-01-1.3335140e-02j,  3.6025030e-02-3.9678380e-02j],
           [ 1.1312112e-01-1.0838963e-01j, -6.0658300e-03+9.9545310e-02j]]]],

           [[[[ 2.4283579e-01-9.0760050e-02j, -1.0328510e-02-3.3271300e-02j],
           [ 8.0414470e-02-7.7696450e-02j, -1.1043184e-01-4.5206200e-03j]],
           [[-2.4174312e-01-1.8906695e-01j,  1.3138287e-01-8.9676440e-02j],
           [ 9.2512020e-02+4.2839120e-02j, -1.0084172e-01+5.7727900e-03j]]],

           [[[ 2.3546338e-01+1.2727812e-01j, -1.6916180e-01-4.5218850e-02j],
           [ 3.8430770e-02-6.4364610e-02j, -4.8739000e-04-4.8112800e-02j]],
           [[ 1.9702384e-01+1.0416970e-02j,  1.7376518e-01+4.0003110e-02j],
           [ 5.9403440e-02-7.7407540e-02j,  1.3129343e-01-1.6172576e-01j]]]]]

true_actions = [[ 2,  1,  2,  0,  2, 19,  3,  0,  2,  3,  2,  0,  5, 15,  6, 10, 15],                   # q0
                [ 5,  4,  6,  7,  5, 12,  4,  6,  4,  6,  7,  6,  4,  5,  1, 15,  2, 10, 15],           # q1
                [12, 10,  8, 11, 10,  8, 10, 11, 10,  8,  9, 10, 11,  0, 15,  2,  6, 15],               # q2
                [15,  3, 12, 14,  0, 13, 12,  5, 15, 13, 12, 15, 13,  0, 11,  1,  5, 11],               # q3
                [12,  3, 18, 14, 19, 16, 18, 19, 17, 16, 18, 19, 17,  0, 10,  1,  5, 10],               # q4
                [12,  3, 11,  7, 15,  0, 17,  9, 12,  0, 10, 19,  6, 10, 15,  3,  7,  8, 16, 12,  1]]   # all

# true_actions = torch.IntTensor(true_actions)

env = QubitsEnvironment(num_qubits=5, epsi=1e-3, batch_size=1)
expert = SearchExpert_OLD(env)

psi = np.array(psi, dtype=np.complex64)
paths = expert.produce_trajectories(psi, beam_size=100)

match = True
for p1, p2 in zip(paths, true_actions):
    if p1 != p2:
        match = False
        print("Oops! Different paths...")
        print("    expert path:", p1)
        print("    true path:  ", p2)

if match:
    print("Trajectories match!")

#
