import os
import sys
sys.path.append("..")

import numpy as np
import torch
import torch.nn.functional as F

from src.envs.rdm_environment import QubitsEnvironment
from src.agents.pg_agent import PGAgent
from src.infrastructure.util_funcs import set_printoptions

set_printoptions(precision=4, sci_mode=False)
log_dir = "../logs/5qubits/test/rollout_test"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


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


# Consider 7 different trajectories. Trajectories from 1 to 5 disentangle one of the
# qubits (1-5 respectively) of the system and then apply a general solution to the
# remaining 4 qubits. Trajectory 6 disentangles all qubits simultaneously.
# And trajectory 7 is a dummy trajectory.
true_actions = [[ 2,  1,  2,  1,  0,  1,  2,  3,  1,  0,  3,  1,  0,  2,  5, 15,  6, 10, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # q0        -110.79
                [ 5,  4,  6,  7,  5,  6,  4,  6,  5,  7,  6,  5,  6,  4,  5,  1, 15,  2, 10, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # q1        -117.61
                [ 9,  4,  9, 10,  9,  8,  9, 10,  9,  8,  9, 11, 10, 11,  0, 15,  2,  6, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # q2        -110.86
                [12, 13, 14, 12, 13, 12, 14, 13, 15, 14, 12, 13, 14, 12, 14, 15,  0, 11,  1,  5, 11,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # q3        -124.54
                [ 5, 17, 19, 18, 17, 16, 19, 18, 16, 19, 17, 18, 19, 18, 19, 17,  0, 10,  1,  5, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0],   # q4        -123.65
                [19,  2, 10,  6, 15, 11,  3, 17,  9,  4, 12,  3, 18,  0,  5, 10, 16, 15, 16,  0,  9,  2,  6, 19,  1, 10,  0,  0,  0,  0],   # all        -80.75
                [11, 12, 15,  3,  3,  6,  1,  2, 18, 16,  1,  2, 10,  0,  7,  8, 15, 16,  9,  8, 14, 19, 16, 11, 17,  9,  9, 15,  3, 10]]   # dummy


# The rewards received by the environment are a living reward `l` and a reward for
# disentangling the system `w`.
l = 0.1
w = 100
true_rewards = [[  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l, w-l, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l, w-l, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l, w-l, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l, w-l, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l, w-l, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l, w-l, 0.0, 0.0, 0.0, 0.0],
                [  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l,  -l]]

true_returns_to_go = [[        w-19*l,        w-18*l,        w-17*l,        w-16*l,        w-15*l,        w-14*l,        w-13*l,        w-12*l,        w-11*l,        w-10*l,         w-9*l,         w-8*l,         w-7*l,         w-6*l,         w-5*l,         w-4*l,         w-3*l,         w-2*l,           w-l,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0],
                      [        w-20*l,        w-19*l,        w-18*l,        w-17*l,        w-16*l,        w-15*l,        w-14*l,        w-13*l,        w-12*l,        w-11*l,        w-10*l,         w-9*l,         w-8*l,         w-7*l,         w-6*l,         w-5*l,         w-4*l,         w-3*l,         w-2*l,           w-l,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0],
                      [        w-19*l,        w-18*l,        w-17*l,        w-16*l,        w-15*l,        w-14*l,        w-13*l,        w-12*l,        w-11*l,        w-10*l,         w-9*l,         w-8*l,         w-7*l,         w-6*l,         w-5*l,         w-4*l,         w-3*l,         w-2*l,           w-l,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0],
                      [        w-21*l,        w-20*l,        w-19*l,        w-18*l,        w-17*l,        w-16*l,        w-15*l,        w-14*l,        w-13*l,        w-12*l,        w-11*l,        w-10*l,         w-9*l,         w-8*l,         w-7*l,         w-6*l,         w-5*l,         w-4*l,         w-3*l,         w-2*l,           w-l,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0],
                      [        w-21*l,        w-20*l,        w-19*l,        w-18*l,        w-17*l,        w-16*l,        w-15*l,        w-14*l,        w-13*l,        w-12*l,        w-11*l,        w-10*l,         w-9*l,         w-8*l,         w-7*l,         w-6*l,         w-5*l,         w-4*l,         w-3*l,         w-2*l,           w-l,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0],
                      [        w-26*l,        w-25*l,        w-24*l,        w-23*l,        w-22*l,        w-21*l,        w-20*l,        w-19*l,        w-18*l,        w-17*l,        w-16*l,        w-15*l,        w-14*l,        w-13*l,        w-12*l,        w-11*l,        w-10*l,         w-9*l,         w-8*l,         w-7*l,         w-6*l,         w-5*l,         w-4*l,         w-3*l,         w-2*l,           w-l,           0.0,           0.0,           0.0,           0.0],
                      [         -30*l,         -29*l,         -28*l,         -27*l,         -26*l,         -25*l,         -24*l,         -23*l,         -22*l,         -21*l,         -20*l,         -19*l,         -18*l,         -17*l,         -16*l,         -15*l,         -14*l,         -13*l,         -12*l,         -11*l,         -10*l,          -9*l,          -8*l,          -7*l,          -6*l,          -5*l,          -4*l,          -3*l,          -2*l,            -l]]


# The baseline is computed as the sum of the returns-to-go divided by the number of
# active trajectories. If only one trajectory is active, then the baseline is 0.
true_baselines =      [ (6*w-156*l)/7, (6*w-149*l)/7, (6*w-142*l)/7, (6*w-135*l)/7, (6*w-128*l)/7, (6*w-121*l)/7, (6*w-114*l)/7, (6*w-107*l)/7, (6*w-100*l)/7,  (6*w-93*l)/7,  (6*w-86*l)/7,  (6*w-79*l)/7,  (6*w-72*l)/7,  (6*w-65*l)/7,  (6*w-58*l)/7,  (6*w-51*l)/7,  (6*w-44*l)/7,  (6*w-37*l)/7,  (6*w-30*l)/7,  (4*w-23*l)/5,  (3*w-18*l)/4,    (w-14*l)/2,    (w-12*l)/2,    (w-10*l)/2,     (w-8*l)/2,     (w-6*l)/2,           0.0,           0.0,           0.0,           0.0]

b = (6*w-156*l)/7
true_baselines_v2 =  [[       19*b/19,       18*b/19,       17*b/19,       16*b/19,       15*b/19,       14*b/19,       13*b/19,       12*b/19,       11*b/19,       10*b/19,        9*b/19,        8*b/19,        7*b/19,        6*b/19,        5*b/19,        4*b/19,        3*b/19,        2*b/19,          b/19,             0,             0,             0,             0,             0,             0,             0,             0,             0,             0,             0],
                      [       20*b/20,       19*b/20,       18*b/20,       17*b/20,       16*b/20,       15*b/20,       14*b/20,       13*b/20,       12*b/20,       11*b/20,       10*b/20,        9*b/20,        8*b/20,        7*b/20,        6*b/20,        5*b/20,        4*b/20,        3*b/20,        2*b/20,          b/20,             0,             0,             0,             0,             0,             0,             0,             0,             0,             0],
                      [       19*b/19,       18*b/19,       17*b/19,       16*b/19,       15*b/19,       14*b/19,       13*b/19,       12*b/19,       11*b/19,       10*b/19,        9*b/19,        8*b/19,        7*b/19,        6*b/19,        5*b/19,        4*b/19,        3*b/19,        2*b/19,          b/19,             0,             0,             0,             0,             0,             0,             0,             0,             0,             0,             0],
                      [       21*b/21,       20*b/21,       19*b/21,       18*b/21,       17*b/21,       16*b/21,       15*b/21,       14*b/21,       13*b/21,       12*b/21,       11*b/21,       10*b/21,        9*b/21,        8*b/21,        7*b/21,        6*b/21,        5*b/21,        4*b/21,        3*b/21,        2*b/21,          b/21,             0,             0,             0,             0,             0,             0,             0,             0,             0],
                      [       21*b/21,       20*b/21,       19*b/21,       18*b/21,       17*b/21,       16*b/21,       15*b/21,       14*b/21,       13*b/21,       12*b/21,       11*b/21,       10*b/21,        9*b/21,        8*b/21,        7*b/21,        6*b/21,        5*b/21,        4*b/21,        3*b/21,        2*b/21,          b/21,             0,             0,             0,             0,             0,             0,             0,             0,             0],
                      [       26*b/26,       25*b/26,       24*b/26,       23*b/26,       22*b/26,       21*b/26,       20*b/26,       19*b/26,       18*b/26,       17*b/26,       16*b/26,       15*b/26,       14*b/26,       13*b/26,       12*b/26,       11*b/26,       10*b/26,        9*b/26,        8*b/26,        7*b/26,        6*b/26,        5*b/26,        4*b/26,        3*b/26,        2*b/26,          b/26,             0,             0,             0,             0],
                      [       30*b/30,       29*b/30,       28*b/30,       27*b/30,       26*b/30,       25*b/30,       24*b/30,       23*b/30,       22*b/30,       21*b/30,       20*b/30,       19*b/30,       18*b/30,       17*b/30,       16*b/30,       15*b/30,       14*b/30,       13*b/30,       12*b/30,       11*b/30,       10*b/30,        9*b/30,        8*b/30,        7*b/30,        6*b/30,        5*b/30,        4*b/30,        3*b/30,        2*b/30,          b/30]]


# A second set of baselines is computed without the dummy trajectory.
true_baselines_mod =  [ (6*w-126*l)/6, (6*w-120*l)/6, (6*w-114*l)/6, (6*w-108*l)/6, (6*w-102*l)/6,  (6*w-96*l)/6,  (6*w-90*l)/6,  (6*w-84*l)/6,  (6*w-78*l)/6,  (6*w-72*l)/6,  (6*w-66*l)/6,  (6*w-60*l)/6,  (6*w-54*l)/6,  (6*w-48*l)/6,  (6*w-42*l)/6,  (6*w-36*l)/6,  (6*w-30*l)/6,  (6*w-24*l)/6,  (6*w-18*l)/6,  (4*w-12*l)/4,   (3*w-8*l)/3,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0,           0.0]

b = (6*w-126*l)/6
true_baselines_mod_v2 =  [[   19*b/19,       18*b/19,       17*b/19,       16*b/19,       15*b/19,       14*b/19,       13*b/19,       12*b/19,       11*b/19,       10*b/19,        9*b/19,        8*b/19,        7*b/19,        6*b/19,        5*b/19,        4*b/19,        3*b/19,        2*b/19,          b/19,             0,             0,             0,             0,             0,             0,             0,             0,             0,             0,             0],
                      [       20*b/20,       19*b/20,       18*b/20,       17*b/20,       16*b/20,       15*b/20,       14*b/20,       13*b/20,       12*b/20,       11*b/20,       10*b/20,        9*b/20,        8*b/20,        7*b/20,        6*b/20,        5*b/20,        4*b/20,        3*b/20,        2*b/20,          b/20,             0,             0,             0,             0,             0,             0,             0,             0,             0,             0],
                      [       19*b/19,       18*b/19,       17*b/19,       16*b/19,       15*b/19,       14*b/19,       13*b/19,       12*b/19,       11*b/19,       10*b/19,        9*b/19,        8*b/19,        7*b/19,        6*b/19,        5*b/19,        4*b/19,        3*b/19,        2*b/19,          b/19,             0,             0,             0,             0,             0,             0,             0,             0,             0,             0,             0],
                      [       21*b/21,       20*b/21,       19*b/21,       18*b/21,       17*b/21,       16*b/21,       15*b/21,       14*b/21,       13*b/21,       12*b/21,       11*b/21,       10*b/21,        9*b/21,        8*b/21,        7*b/21,        6*b/21,        5*b/21,        4*b/21,        3*b/21,        2*b/21,          b/21,             0,             0,             0,             0,             0,             0,             0,             0,             0],
                      [       21*b/21,       20*b/21,       19*b/21,       18*b/21,       17*b/21,       16*b/21,       15*b/21,       14*b/21,       13*b/21,       12*b/21,       11*b/21,       10*b/21,        9*b/21,        8*b/21,        7*b/21,        6*b/21,        5*b/21,        4*b/21,        3*b/21,        2*b/21,          b/21,             0,             0,             0,             0,             0,             0,             0,             0,             0],
                      [       26*b/26,       25*b/26,       24*b/26,       23*b/26,       22*b/26,       21*b/26,       20*b/26,       19*b/26,       18*b/26,       17*b/26,       16*b/26,       15*b/26,       14*b/26,       13*b/26,       12*b/26,       11*b/26,       10*b/26,        9*b/26,        8*b/26,        7*b/26,        6*b/26,        5*b/26,        4*b/26,        3*b/26,        2*b/26,          b/26,             0,             0,             0,             0]]



true_mask = [[  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False],
              [  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False],
              [  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False, False, False],
              [  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False],
              [  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False, False, False, False, False, False],
              [  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True, False, False, False, False],
              [  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True,  True]]

true_nsteps = [19, 20, 19, 21, 21, 26, 0]

# The dummy logits score each action from the action space with its respective index+1.
# action[k] = `qi-qj` ---> score(action[k]) = k+1
dummy_logits = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
dummy_probs = [3.5416428223985797e-09, 9.627183327018465e-09,
            2.6169397497078187e-08, 7.113579767802925e-08,
            1.9336714618112607e-07, 5.256263996851388e-07,
            1.428800690822464e-06,  3.883882954352435e-06,
            1.0557488458678055e-05, 2.8698229031390653e-05,
            7.800987448498504e-05,  0.00021205282425290574,
            0.0005764193388400931,  0.0015668702143414024,
            0.004259194831197964,   0.011577691913512114,
            0.03147142954399722,    0.08554821504507676,
            0.23254415841413886,    0.6321205601314552]

loss = 10.688352530545117
loss_no_dummy = 101.22429074188456

true_actions = torch.IntTensor(true_actions)
true_rewards = torch.FloatTensor(true_rewards)
true_mask = torch.FloatTensor(true_mask)
true_returns_to_go = torch.FloatTensor(true_returns_to_go)
true_baselines_dummy = torch.FloatTensor(true_baselines_v2)
true_baselines_no_dummy = torch.FloatTensor(true_baselines_mod_v2)
dummy_logits = torch.FloatTensor(dummy_logits)
dummy_nll = torch.FloatTensor(dummy_probs)
true_nsteps = torch.IntTensor(true_nsteps)

true_nll = torch.zeros(size=(7,30))
for i, traject in enumerate(true_actions):
    for j, a in enumerate(traject):
        true_nll[i,j] = -np.log(dummy_nll[a])


# Run the tests two times. The first time consider all trajectories, and the second time
# omit the dummy trajectory.
B, steps = true_actions.shape
true_baselines = [true_baselines_dummy, true_baselines_no_dummy]
true_loss = [loss, loss_no_dummy]
msg = ["Testing a batch with a dummy trajectory..", "Testing a batch without a dummy trajectory..."]

for idx, batch_size in enumerate([B, B-1]):
    class DummyPolicy:
        def __init__(self):
            self.i = -1
        def get_action(self, x, greedy=False, beta=1):
            self.i += 1
            return true_actions[:batch_size, self.i]
        @property
        def device(self): return torch.device("cpu")

    env = QubitsEnvironment(5, epsi=1e-3, batch_size=batch_size)
    env.states = np.array([psi] * batch_size, dtype=np.complex64)
    policy = DummyPolicy()
    agent = PGAgent(env, policy, log_dir)
    states, actions, rewards, done = agent.rollout(steps=steps)
    mask = agent.generate_mask(done)

    logits = dummy_logits.repeat(batch_size, steps, 1)
    nll = F.cross_entropy(logits.permute(0,2,1), actions, reduction="none")
    returns_to_go = agent.reward_to_go(rewards)
    baselines = agent.reward_baseline(rewards, mask)
    q_values = returns_to_go - baselines
    weighted_nll = torch.mul(mask * nll, q_values)
    loss = torch.mean(torch.sum(weighted_nll, dim=1))

    nsteps = (~mask[:,-1])*torch.sum(mask, axis=1)

    print(msg[idx])
    if (actions == true_actions[:batch_size]).all(): print("  Actions match!")
    else: print("  Actions don't match!")
    if (abs(rewards-true_rewards[:batch_size])<1e-4).all(): print("  Rewards match!")
    else: print("  Rewards don't match!")
    if (mask == true_mask[:batch_size]).all(): print("  Mask matches!")
    else: print("  Mask doesn't match!")
    if (nsteps == true_nsteps[:batch_size]).all(): print("  Number of steps matches!")
    else: print("  Number of steps doesn't match")
    if (abs(returns_to_go-true_returns_to_go[:batch_size])<1e-4).all(): print("  Returns-to-go match!")
    else: print("  Returns-to-go don't match!")
    if (abs(baselines-true_baselines[idx])<1e-4).all(): print("  Baselines match!")
    else: print("  Baselines don't match!")
    if (abs(nll-true_nll[:batch_size])<1e-4).all(): print("  NLLs match!")
    else: print("  NLLs don't match!")
    if (abs(loss-true_loss[idx])<1e-4): print("  Loss matches!")
    else: print("  Loss doesn't match!")

#