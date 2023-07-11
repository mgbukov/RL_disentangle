import itertools
import functools
import numpy as np
import torch
import matplotlib.pyplot as plt
import ipywidgets as widgets

from copy import deepcopy
from IPython.display import display
from scipy.linalg import expm

from src.quantum_env import rdm_2q_mean_real
from src.quantum_state import (
    VectorQuantumState as RL_Environment,
    random_quantum_state,
    entropy as entanglement
)


SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)


def get_special_states():
    """Returns dictionaty with special states for 4 & 5 qubits."""

    # 2-qubits states
    bell = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]).astype(np.complex64)
    w = np.array([
        0,
        1/np.sqrt(3),
        1/np.sqrt(3),
        0,
        1/np.sqrt(3),
        0,
        0,
        0]).astype(np.complex64)
    # Single qubit states
    zero = np.array([1, 0]).astype(np.complex64)

    rnd_state = functools.partial(random_quantum_state, prob=1.)

    special_states = {
        4: {
            "|0>|0>|0>|0>": np.kron(zero, np.kron(zero, np.kron(zero, zero))),
            "|BB>|0>|0>"  : np.kron(bell, np.kron(zero, zero)),
            "|0>|BB>|0>"  : np.kron(zero, np.kron(bell, zero)),
            "|0>|0>|BB>"  : np.kron(zero, np.kron(zero, bell)),
            "|BB>|R>|R>"  : np.kron(bell, np.kron(rnd_state(q=1), rnd_state(q=1))),
            "|WWW>|0>"    : np.kron(w, zero),
            "|WWW>|R>"    : np.kron(w, rnd_state(q=1)),
            "|RR>|R>|R>"  : np.kron(rnd_state(q=2), np.kron(rnd_state(q=1), rnd_state(q=1))),
            "|RR>|RR>"    : np.kron(rnd_state(q=2), rnd_state(q=2)),
            "|RRR>|0>"    : np.kron(rnd_state(q=3), zero),
            "|RRR>|R>"    : np.kron(rnd_state(q=3), rnd_state(q=1)),
            "|RRRR>"      : rnd_state(q=4),
            "|RRRR>_1"    : rnd_state(q=4),
            "|RRRR>_2"    : rnd_state(q=4),
            "|RRRR>_3"    : rnd_state(q=4),
            "|RRRR>_4"    : rnd_state(q=4),
        },
        5: {
            "|0>|0>|0>|0>|0>": np.kron(zero,np.kron(zero, np.kron(zero, np.kron(zero, zero)))),
            "|BB>|0>|0>|0>": np.kron(bell, np.kron(zero, np.kron(zero, zero))),
            "|0>|BB>|0>|0>": np.kron(zero, np.kron(bell, np.kron(zero, zero))),
            "|0>|0>|BB>|0>": np.kron(zero, np.kron(zero, np.kron(bell, zero))),
            "|0>|0>|0>|BB>": np.kron(zero, np.kron(zero, np.kron(zero, bell))),
            "|BB>|R>|R>|R>": np.kron(bell, np.kron(rnd_state(q=1), np.kron(rnd_state(q=1), rnd_state(q=1)))),
            "|BB>|BB>|0>"  : np.kron(bell, np.kron(bell, zero)),
            "|WWW>|0>|0>"  : np.kron(w, np.kron(zero, zero)),
            "|WWW>|BB>"    : np.kron(w, bell),
            "|RR>|0>|0>|0>": np.kron(rnd_state(q=2), np.kron(zero, np.kron(zero, zero))),
            "|RR>|R>|R>|R>": np.kron(rnd_state(q=2), np.kron(rnd_state(q=1), np.kron(rnd_state(q=1), rnd_state(q=1)))),
            "|RR>|RR>|0>"  : np.kron(rnd_state(q=2), np.kron(rnd_state(q=2), zero)),
            "|RR>|RR>|R>"  : np.kron(rnd_state(q=2), np.kron(rnd_state(q=2), rnd_state(q=1))),
            "|RRR>|0>|0>"  : np.kron(rnd_state(q=3), np.kron(zero, zero)),
            "|RRR>|R>|R>"  : np.kron(rnd_state(q=3), np.kron(rnd_state(q=1), rnd_state(q=1))),
            "|RRR>|RR>"    : np.kron(rnd_state(q=3), rnd_state(q=2)),
            "|RRRR>|0>"    : np.kron(rnd_state(q=4), zero),
            "|RRRR>|R>"    : np.kron(rnd_state(q=4), rnd_state(q=1)),
            "|RRRRR>"      : rnd_state(q=5),
            "|RRRRR>_2"    : rnd_state(q=5),
            "|RRRRR>_3"    : rnd_state(q=5),
            "|RRRRR>_4"    : rnd_state(q=5),
            "|RRRRR>_5"    : rnd_state(q=5),
        }
    }
    return special_states


def unitary_for_nqubits(U, nqubits=1, index=0):
    assert index < nqubits
    I = np.eye(2, 2).astype(np.complex64)
    return functools.reduce(
        np.kron,
        [U if i == index else I for i in range(nqubits)]
    )

def Rx(theta, nqubits=1, index=0):
    rx = np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ]).astype(np.complex64)
    return unitary_for_nqubits(rx, nqubits, index)

def Ry(theta, nqubits=1, index=0):
    ry = np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]]
    ).astype(np.complex64)
    return unitary_for_nqubits(ry, nqubits, index)

def Rz(phi, nqubits=1, index=0):
    rz = np.array([
        [np.exp(0.5j * phi), 0.0 + 0j],
        [0.0 + 0j, np.exp(0.5j * phi)]]
    ).astype(np.complex64)
    return unitary_for_nqubits(rz, nqubits, index)


class SingleQubitRotationControls:

    def __init__(self, num_qubits, update_callback):
        self.num_qubits = int(num_qubits)
        self.update_callback = update_callback
        self.statedict = {}     # i => phi, theta
        for q in range(self.num_qubits):
            axis = np.array([0.0, 0.0], dtype=np.float32)
            angle = 0.0
            self.statedict[q] = (axis, angle)
        # Initialize widgets
        self.angle_silder = widgets.FloatSlider(
            value=0.0, min=0.0, max=2*np.pi, step=0.01,
            description='Angle:', readout_format='.2f',
        )
        self.axis_phi_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=2*np.pi, step=0.01,
            description='Axis Phi:', readout_format='.2f',
        )
        self.axis_theta_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=np.pi, step=0.01,
            description='Axis Theta:', readout_format='.2f',
        )
        self.qubit_selector = widgets.Dropdown(
            options=list(range(self.num_qubits)), value=0,
            description='Qubit:', disabled=False,
        )
        self.handlers = {
            "qubit_selector": self._qubit_selector_callback,
            "angle_slider": self._angle_callback,
            "axis_phi_slider": self._axis_phi_callback,
            "axis_theta_slider": self._axis_theta_callback
        }
        self.observe_all()
        # Initialize layout
        self.layout = widgets.GridspecLayout(3, 3)
        self.layout[0,:] = widgets.HTML(
            value="<center><b>Single Qubit Rotation</b></center>")
        self.layout[1,0] = self.qubit_selector
        self.layout[2,0] = self.axis_phi_slider
        self.layout[2,1] = self.axis_theta_slider
        self.layout[2,2] = self.angle_silder

    def display(self):
        display(self.layout)

    def reset(self):
        # Change all sliders without triggering callback
        self.unobserve_all()
        self.qubit_selector.value = 0
        self.angle_silder.value = 0.0
        self.axis_phi_slider.value = 0.0
        self.axis_theta_slider.value = 0.0
        for q in range(self.num_qubits):
            self.statedict[q] = (np.array([0, 0], dtype=np.float32), 0.0)
        self.observe_all()

    def unobserve_all(self):
        self.angle_silder.unobserve(self.handlers["angle_slider"], 'value')
        self.axis_phi_slider.unobserve(self.handlers["axis_phi_slider"], 'value')
        self.axis_theta_slider.unobserve(self.handlers["axis_theta_slider"], 'value')
        self.qubit_selector.unobserve(self.handlers["qubit_selector"], "value")

    def observe_all(self):
        self.angle_silder.observe(self.handlers["angle_slider"], 'value')
        self.axis_phi_slider.observe(self.handlers["axis_phi_slider"], 'value')
        self.axis_theta_slider.observe(self.handlers["axis_theta_slider"], 'value')
        self.qubit_selector.observe(self.handlers["qubit_selector"], "value")

    def get_state(self):
        return {
            "angle_slider": self.angle_silder.value,
            "axis_phi_slider": self.axis_phi_slider.value,
            "axis_theta_slider": self.axis_theta_slider.value,
            "qubit_selector": self.qubit_selector.value,
            "statedict": deepcopy(self.statedict)
        }

    def set_state(self, state):
        self.unobserve_all()
        self.qubit_selector.value = state["qubit_selector"]
        self.angle_silder.value = state["angle_slider"]
        self.axis_phi_slider.value = state["axis_phi_slider"]
        self.axis_theta_slider.value = state["axis_theta_slider"]
        self.statedict = deepcopy(state["statedict"])
        self.observe_all()

    def _qubit_selector_callback(self, change):
        val = change['new']
        axis, angle = self.statedict[val]
        # Change all sliders without triggering callback
        self.unobserve_all()
        self.angle_silder.value = angle
        self.axis_phi_slider.value = axis[0]
        self.axis_theta_slider.value = axis[1]
        self.observe_all()

    def _angle_callback(self, change):
        val = change['new'] - change['old']
        q = self.qubit_selector.value
        axis, _ = self.statedict[q]
        self.statedict[q] = (axis, change['new'])
        self.update_callback({"qubit_index": q, "axis_phi": axis[0],
                              "axis_theta": axis[1], "angle": val})

    def _axis_phi_callback(self, change):
        q = self.qubit_selector.value
        # Change phi value without triggering callback
        self.angle_silder.unobserve(self.handlers["angle_slider"], "value")
        self.angle_silder.value = 0.0
        self.angle_silder.observe(self.handlers["angle_slider"], "value")
        axis, _ = self.statedict[q]
        axis[0] = change['new']
        self.statedict[q] = (axis, 0.0)

    def _axis_theta_callback(self, change):
        q = self.qubit_selector.value
        # Change theta value without triggering callback
        self.angle_silder.unobserve(self.handlers["angle_slider"], "value")
        self.angle_silder.value = 0.0
        self.angle_silder.observe(self.handlers["angle_slider"], "value")
        axis, _ = self.statedict[q]
        axis[1] = change['new']
        self.statedict[q] = (axis, 0.0)


class TwoQubitsRotationControls:

    def __init__(self, num_qubits, update_callback):
        self.num_qubits = num_qubits
        self.update_callback = update_callback
        self.qubit_indices = list(itertools.combinations(range(num_qubits), 2))
        self.statedict = {}     # (i,j) => alpha, beta, gamma
        for i, j in self.qubit_indices:
            self.statedict[(i,j)] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Initialize widgets
        self.qubits_selector = widgets.Dropdown(
            options=[str(x) for x in self.qubit_indices], value=str((0, 1)),
            description='Qubits:', disabled=False,
        )
        self.alpha_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=2*np.pi, step=0.01,
            description='Alpha:', readout_format='.2f'
        )
        self.beta_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=2*np.pi, step=0.01,
            description='Beta:', readout_format='.2f'
        )
        self.gamma_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=2*np.pi, step=0.01,
            description='Gamma:', readout_format='.2f'
        )
        self.handlers = {
            "qubits_selector": self._qubits_selector_callback,
            "alpha_slider": self._alpha_callback,
            "beta_slider": self._beta_callback,
            "gamma_slider": self._gamma_callback
        }
        self.observa_all()
        # Initialize layout
        self.layout = widgets.GridspecLayout(3, 3)
        self.layout[0,:] = widgets.HTML(
            value="<center><b>Two Qubits Rotation</b></center>")
        self.layout[1,0] = self.qubits_selector
        self.layout[2,0] = self.alpha_slider
        self.layout[2,1] = self.beta_slider
        self.layout[2,2] = self.gamma_slider

    def observa_all(self):
        self.qubits_selector.observe(self.handlers["qubits_selector"], 'value')
        self.alpha_slider.observe(self.handlers["alpha_slider"], 'value')
        self.beta_slider.observe(self.handlers["beta_slider"], 'value')
        self.gamma_slider.observe(self.handlers["gamma_slider"], 'value')

    def unobserve_all(self):
        self.qubits_selector.unobserve(self.handlers["qubits_selector"], 'value')
        self.alpha_slider.unobserve(self.handlers["alpha_slider"], 'value')
        self.beta_slider.unobserve(self.handlers["beta_slider"], 'value')
        self.gamma_slider.unobserve(self.handlers["gamma_slider"], 'value')

    def reset(self):
        # Change all sliders without triggering callback
        self.unobserve_all()
        self.qubits_selector.value = str((0, 1))
        self.alpha_slider.value = 0.0
        self.beta_slider.value = 0.0
        self.gamma_slider.value = 0.0
        for qubits in self.qubit_indices: 
            self.statedict[qubits] = np.zeros(3, dtype=np.float32)
        self.observa_all()

    def display(self):
        display(self.layout)

    def _alpha_callback(self, change):
        i, j = eval(self.qubits_selector.value)
        angles = self.statedict[(i,j)]
        angles[0] = change['new']
        val = change['new'] - change['old']
        self.update_callback({
            "qubit0": i, "qubit1": j, "alpha": val, "beta": 0.0, "gamma": 0.0
        })

    def _beta_callback(self, change):
        i, j = eval(self.qubits_selector.value)
        angles = self.statedict[(i,j)]
        angles[1] = change['new']
        val = change['new'] - change['old']
        self.update_callback({
            "qubit0": i, "qubit1": j, "alpha": 0.0, "beta": val, "gamma": 0.0
        })

    def _gamma_callback(self, change):
        i, j = eval(self.qubits_selector.value)
        angles = self.statedict[(i,j)]
        angles[2] = change['new']
        val = change['new'] - change['old']
        self.update_callback({
            "qubit0": i, "qubit1": j, "alpha": 0.0, "beta": 0.0, "gamma": val
        })

    def _qubits_selector_callback(self, change):
        val = eval(change['new'])
        angles = self.statedict[val]
        # Change all sliders without triggering callback
        self.unobserve_all()
        self.alpha_slider.value = angles[0]
        self.beta_slider.value = angles[1]
        self.gamma_slider.value = angles[2]
        self.observa_all()

    def get_state(self):
        return {
            "qubits_selector": self.qubits_selector.value,
            "alpha_slider": self.alpha_slider.value,
            "beta_slider": self.beta_slider.value,
            "gamma_slider": self.gamma_slider.value,
            "statedict": deepcopy(self.statedict)
        }

    def set_state(self, state):
        self.unobserve_all()
        self.qubits_selector.value = state["qubits_selector"]
        self.alpha_slider.value = state["alpha_slider"]
        self.beta_slider.value = state["beta_slider"]
        self.gamma_slider.value = state["gamma_slider"]
        self.statedict = deepcopy(state["statedict"])
        self.observa_all()


class QuantumState:

    def __init__(self, state, handler=None):
        self.num_qubits = int(np.log2(state.size))
        self.shape = (1,) + (2,) * self.num_qubits
        assert state.size == 2 ** self.num_qubits
        self._vector = np.zeros(2 ** self.num_qubits, dtype=np.complex64)
        self.handler = handler
        # Initialize widgets
        self.entanglements_status = widgets.HTML(value="<b>Entanglements</b: ")
        self.set_button = widgets.Button(
            description="Set",
            layout=widgets.Layout(height='auto', width='auto')
        )
        self.set_button.on_click(self._set_callback)
        # Initialize layout
        self.layout = widgets.GridspecLayout(4, 8)
        self.layout[0, :6] = self.entanglements_status
        self.layout[1, :4] = widgets.HTML(value="<b>Vector Components:</b>")
        self.layout[1, 6:] = self.set_button
        for i, j in itertools.product(range(2,4), range(8)):
            self.layout[i,j] = widgets.Text(
                value='', layout=widgets.Layout(width='110px')
            )
        # Assign vector state
        self.vector = state

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, newstate):
        self._vector = newstate.ravel()
        self.update_entanglements()
        # Update textboxes
        n = 0
        for i, j in itertools.product(range(2,4), range(8)):
            x = self._vector[n]
            sign = '-' if x.imag < 0.0 else '+'
            self.layout[i,j].value = f"{x.real:.3f} {sign}{abs(x.imag):.3f}j"
            n += 1
        self.handler(self._vector)

    def display(self):
        display(self.layout)

    def update_entanglements(self):
        self.entanglements = entanglement(self._vector.reshape(self.shape))[0]
        html_value = (4 * "&nbsp").join(f"{x:.3f}" for x in self.entanglements)
        msg = "<b>Entanglements:</b> " + 4 * "&nbsp" + html_value
        self.entanglements_status.value = msg

    def _set_callback(self, b):
        # Read values from textboxes
        values = []
        for i, j in itertools.product(range(2,4), range(8)):
            try:
                x = complex(self.layout[i,j].value.replace(" ", ""))
            except ValueError:
                # Use old value
                x = self._vector[8*(i-2) + j]
            values.append(x)
        values = np.asarray(values, dtype=np.complex64)
        values /= np.linalg.norm(values)
        self.vector = values


def apply_1q_rotation(qstate, info):
    phi, theta = info['axis_phi'], info['axis_theta']
    h = [np.sin(theta) * np.cos(phi),
         np.sin(theta) * np.sin(phi),
         np.cos(theta)]
    M = h[0] * SIGMA_X + h[1] * SIGMA_Y + h[2] * SIGMA_Z
    U = expm(-1.0j * (info["angle"] / 2) * M)
    U = unitary_for_nqubits(U, qstate.num_qubits, info["qubit_index"])
    qstate.vector = np.matmul(U, qstate.vector.reshape(-1, 1))


def apply_2q_rotation(qstate, info):
    psi = qstate.vector
    tshape = (2,) * qstate.num_qubits
    psi = psi.reshape(tshape)
    qubits = [info["qubit0"], info["qubit1"]]
    T = qubits + [k for k in range(qstate.num_qubits) if k not in qubits]
    psi = np.transpose(psi, T)
    sigma_x = np.kron(SIGMA_X, SIGMA_X)
    sigma_y = np.kron(SIGMA_Y, SIGMA_Y)
    sigma_z = np.kron(SIGMA_Z, SIGMA_Z)
    M = info["alpha"] * sigma_x + info["beta"] * sigma_y + info["gamma"] * sigma_z
    U = expm(np.array([-1.0j], dtype=np.complex64) * M)
    psi = np.matmul(U, psi.reshape(4, -1)).reshape(tshape)
    psi = np.transpose(psi, np.argsort(T))
    qstate.vector = psi.ravel()


class InteractiveAttentionScores:

    def __init__(self, num_qubits, agent):
        self.agent = agent
        self.num_qubits = int(num_qubits)
        self.actions = list(itertools.combinations(range(self.num_qubits), 2))
        self._env = RL_Environment(self.num_qubits, 1)

    def set_prob_ax(self, ax):
        self.prob_ax = ax

    def set_attn_axes(self, axs):
        self.attn_axs = axs

    def get_attention_scores(self, state):
        agent = self.agent
        o = rdm_2q_mean_real(state.reshape((1,) + (2,) * self.num_qubits))
        o = torch.from_numpy(o)

        with torch.no_grad():
            emb = agent.policy_network.net[0](o)
            attn_weights = []
            # First layer is the embedding layer, last two layers are output layers
            for i in range(1, len(agent.policy_network.net)-2):
                z_norm = agent.policy_network.net[i].norm1(emb)
                _, attn = agent.policy_network.net[i].self_attn(
                    z_norm, z_norm, z_norm,
                    need_weights=True, average_attn_weights=False
                )
                emb = agent.policy_network.net[i](emb)
                attn_weights.append(attn.numpy())
            attn_weights = np.vstack(attn_weights)
        return attn_weights

    def get_probabilities(self, state):
        state = state.reshape((1,) + (2,) * self.num_qubits)
        o = torch.from_numpy(rdm_2q_mean_real(state))
        pi = self.agent.policy(o)
        return pi.probs[0].numpy()

    def plot_probabilities(self, state):
        probs = self.get_probabilities(state)
        act_taken = np.argmax(probs)
        xs = np.arange(len(probs))
        self.prob_ax.clear()
        self.prob_ax.set_title('Action Probabilities')
        self.prob_ax.set_ylim(0.0, 1.05)
        self.prob_ax.set_xticks(xs, [str(a) for a in self.actions])
        pps = self.prob_ax.bar(xs, probs, alpha=0.5)
        pps[act_taken].set_alpha(1.0)
        return probs

    def plot_entanglement_reduction(self, state, bottom):
        """Should be called after `self.plot_probabilities()`"""
        psi = state.reshape((1,) + (2,) * self.num_qubits)
        reductions = {}
        current_entanglements = entanglement(psi)
        for i, a in enumerate(self.actions):
            self._env.states = psi
            self._env.apply([i])
            new_entanglements = self._env.entanglements
            assert new_entanglements.shape == current_entanglements.shape
            r = current_entanglements.ravel() - new_entanglements.ravel()
            reductions[a] = np.mean(r)
        xs = np.arange(len(self.actions))
        ys = np.array([reductions[a] for a in self.actions])
        self.prob_ax.bar(xs, ys, bottom=bottom, color='tab:red')
        for x, y, b in zip(xs, ys, bottom):
            self.prob_ax.text(
                x, y+b, f"{y:.2f}", horizontalalignment="center",
                verticalalignment="bottom",
                fontsize="small"
            )
        # Update axes ylim
        ymax = np.max(bottom + ys)
        self.prob_ax.set_ylim(0, max(1.05, ymax + 0.05))

    def plot_attention_scores(self, state):
        attention_scores = self.get_attention_scores(state)
        n_layers, n_heads = attention_scores.shape[0], attention_scores.shape[1]
        seq = [f"{i}{j}" for i, j in self.actions]

        axiter = self.attn_axs.flat
        for i, j in itertools.product(range(n_layers), range(n_heads)):
            ax = next(axiter)
            ax.clear()
            ax.imshow(attention_scores[i][j].T, vmin=0., vmax=1.)
            ax.set_title(f"L{i+1} H{j+1}")
            ax.set_xticks(np.arange(len(list(seq))), seq)
            ax.set_yticks(np.arange(len(list(seq))), seq)

    def update(self, state):
        self.plot_attention_scores(state)
        probs = self.plot_probabilities(state)
        self.plot_entanglement_reduction(state, probs)


class StepControls:

    def __init__(self, agent, qstate, controls_to_reset=tuple()):
        self.step_button = widgets.Button(
            description="Step",
            layout=widgets.Layout(height='auto', width='auto')
        )
        self.step_button.on_click(self.step)
        self.undo_button = widgets.Button(
            description="Undo", disabled=True,
            layout=widgets.Layout(height='auto', width='auto')
        )
        self.undo_button.on_click(self.undo)
        self.nsteps_status = widgets.HTML(value="<b>N Steps:</b> 0")
        self.nsteps = 0
        self.qstate = qstate
        self.controls_to_reset = list(controls_to_reset)
        self.agent = agent
        self._env = RL_Environment(qstate.num_qubits, 1)
        self.layout = widgets.GridspecLayout(1, 6)
        self.layout[0,0] = self.nsteps_status
        self.layout[0,2] = self.undo_button
        self.layout[0,3] = self.step_button
        self.history = []
        self._save_controls_state()

    def step(self, b):
        state = self.qstate.vector.reshape((1,) + (2,) * self.qstate.num_qubits)
        self._env.states = state
        self._env.apply([self._get_action(state)])
        # Save the state of dependent controls in self.history. Enables UNDO
        self._save_controls_state()
        self.qstate.vector = self._env.states[0].ravel()
        for ctrl in self.controls_to_reset:
            ctrl.reset()
        self.undo_button.disabled = False
        # Update "N Steps" status
        self.nsteps += 1
        self.nsteps_status.value = f"<b>N Steps:</b> {self.nsteps}"


    def undo(self, b):
        if len(self.history) < 2:
            return False
        # Restore the state of dependent controls from self.history.
        s = self.history.pop()
        for control in self.controls_to_reset:
            control.set_state(s[id(control)])
        self.qstate.vector = s[id(self.qstate)]
        # Update "N Steps" status
        self.nsteps -= 1
        self.nsteps_status.value = f"<b>N Steps:</b> {self.nsteps}"
        if self.nsteps == 0:
            self.undo_button.disabled = True


    def display(self):
        display(self.layout)

    def _save_controls_state(self):
        # Save the state of each control
        s = {}
        for ctrl in self.controls_to_reset:
            s[id(ctrl)] = ctrl.get_state()
        s[id(self.qstate)] = self.qstate.vector
        self.history.append(s)

    def _get_action(self, state):
        state = state.reshape((1,) + (2,) * self.qstate.num_qubits)
        o = torch.from_numpy(rdm_2q_mean_real(state))
        pi = self.agent.policy(o)
        return np.argmax(pi.probs[0].numpy())
