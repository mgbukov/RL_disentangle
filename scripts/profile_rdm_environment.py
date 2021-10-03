import argparse
import cProfile
import datetime
import io
import numpy as np
import pstats
import sys
import os

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.split(PATH)[0])
from src.envs.rdm_environment import QubitsEnvironment


parser = argparse.ArgumentParser()
parser.add_argument('-L', '--L', action='store', help='Number of qubits, default=8', type=int)
parser.add_argument('-B', '--B', action='store', help='Batch size, default=256', type=int)
parser.add_argument('--steps', '-s', action='store', help='Number of rollout steps, default=100', type=int)
parser.add_argument('--output', '-o',
                    action='store_true',
                    help='If set, saves the profile results to textfile in the current directory')
parser.add_argument('--filename', '-f', action='store',
                    help='Filename in which to save profile results')
args = parser.parse_args()


def do_rollout(env, actions):
    for a in actions:
        states, rewards, done = env.step(a)

L = args.L or 8
B = args.B or 256
steps = args.steps or 100

E = QubitsEnvironment(num_qubits=L, batch_size=B)
actions = np.random.uniform(0, E.num_actions, (steps, B)).astype(np.int32)

# Run
print(f'Profiling with L={L}, B={B}, steps={steps}\n...')

P = cProfile.Profile()
P.run('do_rollout(E, actions)')

if args.filename or args.output:
    default = 'rdm-profile-{}.txt'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S'))
    filename = args.filename or default
    print(f'Saving profile results to "{filename}"\n...')
    with open(filename, 'w') as f:
        s = io.StringIO()
        ps = pstats.Stats(P, stream=s).sort_stats('tottime')
        ps.strip_dirs().sort_stats('tottime').print_stats()
        f.write(s.getvalue())
else:
    ps = pstats.Stats(P).sort_stats('tottime')
    ps.strip_dirs().sort_stats('tottime').print_stats()
