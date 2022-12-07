import numpy as np
import matplotlib.pyplot as plt

from common import environment_generator, phase_norm


# Global constants for all tests
EPSI = 1e-3
SEED = 15 #


def test_state_norm():
    """ Tests if rollout presrves state norm. """
    _test_state_norm(nqubits=None, batch_size=64, stochastic=(False, True),
                     nrepeats=10)

def _test_state_norm(nqubits=None, batch_size=None, stochastic=False,
                     nrepeats=10):
    envgen = environment_generator(nqubits, batch_size, stochastic, SEED, EPSI)
    for env in envgen:
        action_set = env.actions
        for _ in range(nrepeats):
            states = env.states.reshape(env.batch_size, -1)
            norms = np.linalg.norm(states, axis=1)
            assert np.all(np.isclose(norms, 1))
            a = np.random.randint(
                size=env.batch_size,low=min(action_set), high=max(action_set)-1)
            env.step(a)


def test_repeated_action():
    """ Tests if repeating the same action multiple times modifies the state."""
    _test_repeated_action(None, 64, 100)

def _test_repeated_action(nqubits=None, batch_size=None, nrepeats=10):
    envgen = environment_generator(nqubits, batch_size, False, SEED, EPSI)
    I = np.eye(4, 4, dtype=np.complex64)
    for env in envgen:
        for a in env.actions.keys():
            env.set_random_states()
            psi, _, _ = env.step([a] * env.batch_size)
            psi = psi.copy()
            for _ in range(nrepeats):
                phi, _, _ = env.step([a] * env.batch_size)
                assert np.all(np.isclose(I, env.unitary[0]))
            A = psi.reshape(env.batch_size, -1)
            C = phi.reshape(env.batch_size, -1)
            overlap = np.sum(np.abs(A.conj() * C), axis=1) ** 2
            assert np.all(np.abs(1 - overlap) < 1e-6)


def test_entropy_cache():
    """ Tests if the entropy cache is updated correctly."""
    _test_entropy_cache(nqubits=None, batch_size=64, stochastic=(False, True),
                        nrepeats=250)

def _test_entropy_cache(nqubits=None, batch_size=None, stochastic=False,
                        nrepeats=10):
    for env in environment_generator(nqubits, batch_size, stochastic, SEED, EPSI):
        action_set = list(env.actions.keys())
        for _ in range(nrepeats):
            actions = np.random.choice(action_set, env.batch_size, True)
            states, _, _ = env.step(actions)
            entropies = env.Entropy(states)
            try:
                assert np.all(np.isclose(entropies, env.entropy(), atol=1e-5))
            except AssertionError as ex:
                print(np.round(env.entropy(), 5))
                print(np.round(entropies, 5))
                raise ex


def test_batch_precision():
    """ Tests if batched rollout results in the same terminal states as single
    state rollouts. """
    _test_batch_precision(4, 16, 10)
    _test_batch_precision(5, 64, 40)
    _test_batch_precision(6, 64, 40)
    _test_batch_precision(7, 64, 120)

def _test_batch_precision(nqubits, batch_size, nsteps, debug=False):
    for env in environment_generator(nqubits, batch_size, False, SEED, EPSI):
        env.set_random_states()
        start_states = env.states.copy()
        action_set = list(env.actions.keys())
        actions = np.random.choice(
            action_set, size=(nsteps, env.batch_size), replace=True)
        # actions = np.random.choice(
        #     action_set, size=nsteps, replace=True
        # )
        # Batched rollout
        batched_rollout = [start_states.copy()]
        for a in actions:
            next_states, _, _ = env.step(a)
            batched_rollout.append(next_states.copy())
        batched_rollout = np.array(batched_rollout)  # (T, B, 2, 2,...)
        # Single state rollout
        single_rollout = []
        for i in range(env.batch_size):
            env.states = np.expand_dims(start_states[i], 0).copy()
            roll = [env.states.copy()[0]]
            # roll = [start_states[i].copy()]
            for j in range(nsteps):
                next_states, _, _ = env.step(actions[j, i])
                # next_states, _, _ = env.step(actions[j])
                roll.append(next_states[0].copy())
            single_rollout.append(np.array(roll))
        single_rollout = np.array(single_rollout)           # (B, T, 2, 2,...)
        single_rollout = np.swapaxes(single_rollout, 0, 1)  # (T, B, 2, 2,...)
        assert single_rollout.shape == batched_rollout.shape
        assert np.all(single_rollout[0] == batched_rollout[0])
        # Compare overlap
        A = batched_rollout.reshape(batch_size, (nsteps + 1), -1)   # (B, T, L)
        B = single_rollout.reshape(batch_size, (nsteps + 1), -1)    # (B, T, L)
        difference = np.abs(A - B)
        overlaps = np.abs(np.sum(A.conj() * B, axis=2)) ** 2
        if debug:
            print('Overlaps:', overlaps)
            print('Minimum overlap:', np.min(overlaps))
            print('Maximum delta:', np.max(difference, axis=2))
            fig, ax = plt.subplots()
            pmesh = ax.pcolormesh(np.max(difference, axis=2))
            ax.set_xlabel('step index')
            ax.set_ylabel('batch index')
            figname = f'difference-{batch_size}-{nqubits}.png'
            figtitle = 'Difference between batched & solo rollouts\n' \
                    f'batch_size={batch_size}, num_qubits={nqubits}'
            ax.set_title(figtitle)
            fig.colorbar(pmesh)
            fig.savefig(figname)
            plt.close(fig)
        assert np.all(np.isclose(overlaps, 1.0, atol=1e-7))


def test_batch_copy_equivalence():
    """ Tests that rollout with single state is equivalent to batched rollout
    with copy of that state."""
    _test_batch_copy_equivalence(4, (1, 2, 8, 64, 256, 1024), 50)
    _test_batch_copy_equivalence(5, (1, 2, 8, 64, 256, 1024), 50)
    _test_batch_copy_equivalence(6, (1, 2, 8, 64, 256, 1024), 50)
    _test_batch_copy_equivalence(7, (1, 2, 8, 64, 256, 1024), 50)
    _test_batch_copy_equivalence(8, (1, 2, 8, 64, 256, 1024), 50)

def _test_batch_copy_equivalence(nqubits=None, bsizes=None, nsteps=50):
    for env in environment_generator(nqubits, bsizes, False, SEED, EPSI):
        B = env.batch_size
        state = env.states[0].copy()
        batch = np.broadcast_to(state, env.shape)
        env.states = batch
        assert env.batch_size == len(batch)
        action_set = list(env.actions.keys())
        actions = np.random.choice(
            action_set, size=nsteps, replace=True)
        # Batched rollout
        batched_rollout = [env.states.copy()]
        for a in actions:
            next_states, _, _ = env.step([a] * B)
            batched_rollout.append(next_states.copy())
        # Solo rollout
        env.states = np.expand_dims(state, 0)
        assert env.batch_size == 1
        solo_rollout = [env.states[0]]
        for a in actions:
            next_states, _, _ = env.step(a)
            solo_rollout.append(next_states[0].copy())
        solo_rollout = np.array(solo_rollout)
        for i in range(len(solo_rollout)):
            b = batched_rollout[i]
            s = np.expand_dims(solo_rollout[i], 0)
            assert np.all(s == b)


def test_phase_norm():
    """Tests if phase_norm() """
    _test_phase_norm(4, (1, 2, 8, 64, 256, 1024), stochastic=(False, True))
    _test_phase_norm(5, (1, 2, 8, 64, 256, 1024), stochastic=(False, True))
    _test_phase_norm(6, (1, 2, 8, 64, 256, 1024), stochastic=(False, True))
    _test_phase_norm(7, (1, 2, 8, 64, 256, 1024), stochastic=(False, True))

def _test_phase_norm(nqubits, bsizes, stochastic=False, repeats=10, debug=False):
    for env in environment_generator(nqubits, bsizes, stochastic, SEED, EPSI):
        start_states = env.states.copy()
        for s in start_states:
            assert s.flat[0].imag == 0.0
        for i in range(repeats):
            states = phase_norm(start_states)
            if debug:
                print(i)
                print(start_states.reshape(env.batch_size, -1), '\n\n',
                      states.reshape(env.batch_size, -1))
            assert np.all(states == start_states)


def test_state_assignment():
    """ Tests the assignment of states to the environment. """



if __name__ == '__main__':
    test_phase_norm()
    test_batch_precision()
    test_batch_copy_equivalence()
    test_state_norm()
    test_repeated_action()
    test_entropy_cache()