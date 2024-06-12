import itertools
import functools
import os

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
import numpy as np
import torch

from copy import deepcopy
from IPython.display import display
from scipy.linalg import expm

from src.quantum_env import rdm_2q_mean_real as observe
from src.quantum_state import (
    random_quantum_state,
    entropy as calc_entanglement,
    VectorQuantumState as QStateSimulator
)

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'


SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex64)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex64)

PROJECT_DIR = os.path.dirname(__file__)
PATH_4Q_AGENT = os.path.join(PROJECT_DIR, "agents", "4q-agent.pt")
PATH_5Q_AGENT = os.path.join(PROJECT_DIR, "agents", "5q-agent.pt")

BUTTON_FONT_SIZE = "11pt"
COLOR0 = "#60b4e1"  # dark blue
COLOR1 = "#d6f0f6"  # light blue


class RotationControls1q:

    def __init__(self, mediator, n_qubits: int):
        self.n_qubits = n_qubits
        self.mediator = mediator
        # We want to enable undo/redo actions for this widget and going back
        # to last slider values whenever the qubit selector dropdown menu
        # changes the index of the qubit. Keeping only the slider values as
        # attributes will not do to implement the specification.
        # Thus, we'll keep the (phi, theta) angles for each qubit
        # in a dictionary
        self.statedict = {}     # i => (phi, theta)
        for q in range(self.n_qubits):
            axis = np.array([0.0, 0.0], dtype=np.float32)
            angle = 0.0
            self.statedict[q] = (axis, angle)
        # Initialize widgets
        self.angle_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=2*np.pi, step=0.01,
            description=r"$\eta$", readout_format='.2f',
        )
        self.angle_slider.style.handle_color = COLOR0
        self.axis_phi_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=2*np.pi, step=0.01,
            description=r"$\text{axis}\ \phi$", readout_format='.2f',
        )
        self.axis_phi_slider.style.handle_color = COLOR0
        self.axis_theta_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=np.pi, step=0.01,
            description=r"$\text{axis}\ \theta$", readout_format='.2f',
        )
        self.axis_theta_slider.style.handle_color = COLOR0
        options = list(range(self.n_qubits))
        self.qselector = widgets.Dropdown(
            options={k+1:k for k in options}, value=0,
            description=r"$\text{qubit}\ i$", disabled=False,
        )
        self._widgets = [self.angle_slider, self.axis_phi_slider,
                         self.axis_theta_slider, self.qselector]
        self._handlers = [self._angle_callback, self._phi_callback,
                          self._theta_callback, self._qselector_callback]

        observe_widgets(self._widgets, self._handlers)

        self.title = widgets.HTML(
            value=f"<b style=\"font-size:{BUTTON_FONT_SIZE}\">Single qubit rotation:</b>")
        imgpath = os.path.join(PROJECT_DIR, "images", "rotation-1q.png")
        if os.path.exists(imgpath):
            with open(imgpath, mode="rb") as f:
                data = f.read()
            self.formula = widgets.Image(value=data, format="png",
                                         layout=dict(height="18px", width="auto"))
        else:
            self.formula = widgets.HTML(value="")
        # Initialize layout
        self.layout = widgets.GridspecLayout(2, 6)
        self.layout[0,0] = self.title
        self.layout[0,1:4] = self.formula
        self.layout[0,4:6] = self.qselector
        self.layout[1,0:2] = self.axis_phi_slider
        self.layout[1,2:4] = self.axis_theta_slider
        self.layout[1,4:6] = self.angle_slider

    def display(self):
        display(self.layout)

    def reset(self):
        # Change all sliders without triggering callback
        unobserve_widgets(self._widgets, self._handlers)
        self.qselector.value = 0
        self.angle_slider.value = 0.0
        self.axis_phi_slider.value = 0.0
        self.axis_theta_slider.value = 0.0
        for q in range(self.n_qubits):
            self.statedict[q] = (np.array([0, 0], dtype=np.float32), 0.0)
        observe_widgets(self._widgets, self._handlers)

    def get_state(self):
        return {
            "angle_slider": self.angle_slider.value,
            "axis_phi_slider": self.axis_phi_slider.value,
            "axis_theta_slider": self.axis_theta_slider.value,
            "qselector": self.qselector.value,
            "statedict": deepcopy(self.statedict)
        }

    def set_state(self, state):
        unobserve_widgets(self._widgets, self._handlers)
        self.qselector.value = state["qselector"]
        self.angle_slider.value = state["angle_slider"]
        self.axis_phi_slider.value = state["axis_phi_slider"]
        self.axis_theta_slider.value = state["axis_theta_slider"]
        self.statedict = deepcopy(state["statedict"])
        observe_widgets(self._widgets, self._handlers)

    def _qselector_callback(self, change):
        val = change['new']
        axis, angle = self.statedict[val]
        # Change all sliders without triggering callback
        unobserve_widgets(self._widgets, self._handlers)
        self.angle_slider.value = angle
        self.axis_phi_slider.value = axis[0]
        self.axis_theta_slider.value = axis[1]
        observe_widgets(self._widgets, self._handlers)

    def _angle_callback(self, change):
        val = change['new'] - change['old']
        q = self.qselector.value
        axis, _ = self.statedict[q]
        self.statedict[q] = (axis, change['new'])
        self.mediator.update(
            "rotate-1q", qubit=q, phi=axis[0], theta=axis[1], angle=val)

    def _phi_callback(self, change):
        q = self.qselector.value
        # Undo the last rotation
        axis, alpha = self.statedict[q]
        self.mediator.update(
            "rotate-1q", qubit=q, phi=axis[0], theta=axis[1], angle=-alpha)
        # Apply the new rotation
        axis[0] = change['new']
        self.statedict[q] = (axis, alpha)
        self.mediator.update(
            "rotate-1q", qubit=q, phi=axis[0], theta=axis[1], angle=alpha)

    def _theta_callback(self, change):
        q = self.qselector.value
        # Undo the last rotation
        axis, alpha = self.statedict[q]
        self.mediator.update(
            "rotate-1q", qubit=q, phi=axis[0], theta=axis[1], angle=-alpha)
        # Apply the new rotation
        axis[1] = change['new']
        self.statedict[q] = (axis, alpha)
        self.mediator.update(
            "rotate-1q", qubit=q, phi=axis[0], theta=axis[1], angle=alpha)


class RotationControls2q:

    def __init__(self, mediator, n_qubits=4):
        self.mediator = mediator
        self.n_qubits = n_qubits
        self.qubit_indices = list(itertools.combinations(range(n_qubits), 2))
        self.statedict = {}     # (i,j) => alpha, beta, gamma
        for i, j in self.qubit_indices:
            self.statedict[(i,j)] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Initialize widgets
        options = {str((i+1,j+1)): (i,j) for i,j in self.qubit_indices}
        self.qselector = widgets.Dropdown(options=options, value=(0, 1),
            description=r"$\text{qubits}\ (i,j)$", disabled=False,
        )
        self.alpha_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=2*np.pi, step=0.01,
            description=r"$\alpha$", readout_format='.2f'
        )
        self.alpha_slider.style.handle_color = COLOR0
        self.beta_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=2*np.pi, step=0.01,
            description=r"$\beta$", readout_format='.2f'
        )
        self.beta_slider.style.handle_color = COLOR0
        self.gamma_slider = widgets.FloatSlider(
            value=0.0, min=0.0, max=2*np.pi, step=0.01,
            description=r"$\gamma$", readout_format='.2f'
        )
        self.gamma_slider.style.handle_color = COLOR0
        self._widgets = (self.qselector, self.alpha_slider, self.beta_slider,
                         self.gamma_slider)
        self._handlers = (self._selector_callback, self._alpha_callback,
                          self._beta_callback, self._gamma_callback)

        observe_widgets(self._widgets, self._handlers)

        self.title = widgets.HTML(
            value=f"<b style=\"font-size:{BUTTON_FONT_SIZE}\">Two-qubit rotation:</b>")
        # Formula
        imgpath = os.path.join(PROJECT_DIR, "images", "rotation-2q.png")
        if os.path.exists(imgpath):
            with open(imgpath, mode="rb") as f:
                data = f.read()
            self.formula = widgets.Image(value=data, format="png",
                                         layout=dict(height="18px", width="auto"))
        else:
            self.formula = widgets.HTML(value="")

        # Initialize layout
        self.layout = widgets.GridspecLayout(2, 6)
        self.layout[0,:1] = self.title
        self.layout[0,1:4] = self.formula
        self.layout[0,4:6] = self.qselector
        self.layout[1,0:2] = self.alpha_slider
        self.layout[1,2:4] = self.beta_slider
        self.layout[1,4:6] = self.gamma_slider

    def reset(self):
        # Change all sliders without triggering callback
        unobserve_widgets(self._widgets, self._handlers)
        self.qselector.value = (0, 1)
        self.alpha_slider.value = 0.0
        self.beta_slider.value = 0.0
        self.gamma_slider.value = 0.0
        for qubits in self.qubit_indices: 
            self.statedict[qubits] = np.zeros(3, dtype=np.float32)
        observe_widgets(self._widgets, self._handlers)

    def get_state(self):
        return {
            "qubits_selector": self.qselector.value,
            "alpha_slider": self.alpha_slider.value,
            "beta_slider": self.beta_slider.value,
            "gamma_slider": self.gamma_slider.value,
            "statedict": deepcopy(self.statedict)
        }

    def set_state(self, state):
        unobserve_widgets(self._widgets, self._handlers)
        self.qselector.value = state["qubits_selector"]
        self.alpha_slider.value = state["alpha_slider"]
        self.beta_slider.value = state["beta_slider"]
        self.gamma_slider.value = state["gamma_slider"]
        self.statedict = deepcopy(state["statedict"])
        observe_widgets(self._widgets, self._handlers)

    def display(self):
        display(self.layout)

    def _alpha_callback(self, change):
        i, j = self.qselector.value
        angles = self.statedict[(i,j)]
        angles[0] = change['new']
        val = change['new'] - change['old']
        self.mediator.update(
            "rotate-2q", qubit0=i, qubit1=j, alpha=val, beta=0.0, gamma=0.0)

    def _beta_callback(self, change):
        i, j = self.qselector.value
        angles = self.statedict[(i,j)]
        angles[1] = change['new']
        val = change['new'] - change['old']
        self.mediator.update(
            "rotate-2q", qubit0=i, qubit1=j, alpha=0.0, beta=val, gamma=0.0)

    def _gamma_callback(self, change):
        i, j = self.qselector.value
        angles = self.statedict[(i,j)]
        angles[2] = change['new']
        val = change['new'] - change['old']
        self.mediator.update(
            "rotate-2q", qubit0=i, qubit1=j, alpha=0.0, beta=0.0, gamma=val)

    def _selector_callback(self, change):
        val = change['new']
        angles = self.statedict[val]
        # Change all sliders without triggering callback
        unobserve_widgets(self._widgets, self._handlers)
        self.alpha_slider.value = angles[0]
        self.beta_slider.value = angles[1]
        self.gamma_slider.value = angles[2]
        observe_widgets(self._widgets, self._handlers)


class AmplitudesBox:

    def __init__(self, mediator=None, n_qubits=4):
        self.mediator = mediator
        self.n_qubits = n_qubits
        self._nrows = 1 + 2 * max(1, (2 ** n_qubits) // 8)

        # Initialize Set button
        self.set_button = widgets.Button(
            description="Set", layout=widgets.Layout(height="auto", width="120px")
        )
        self.set_button.style.font_size = BUTTON_FONT_SIZE
        self.set_button.style.button_color = COLOR1
        self.set_button.on_click(self._set_callback)
        self.title = widgets.HTML(
            value=f"<b style=\"font-size: {BUTTON_FONT_SIZE}\">Computational / z-basis amplitudes:</b>")

        # Initialize Layout
        self.layout = widgets.GridspecLayout(self._nrows+1, 8)
        self.layout[0, :2] = self.title
        self.layout[0, 2:3] = self.set_button

        # Initialize Text boxes & Basis Annotations
        # Skip row 0 - it contains the title and Set button
        # === --------------------------------- ===
        #   Initialize basis annotations
        #   Odd numbered rows will contatin image annotations of basis
        basis = itertools.product(*(("01",) * self.n_qubits))
        self.annotations = []
        for i in range(1, self._nrows, 2):
            for j in range(8):
                b = ''.join(next(basis))
                imgpath = os.path.join(PROJECT_DIR, "images", f"{b}.png")
                if not os.path.exists(imgpath):
                    self.annotations.append(widgets.HTML(value=""))
                    continue
                with open(imgpath, mode="rb") as f:
                    data = f.read()
                annot = widgets.Image(
                    value=data, format="png",
                    layout=dict(height="18px", margin="10px 0px 0px 2px", ))
                self.layout[i, j] = annot
                self.annotations.append(annot)

        #   Initialize text boxes
        #   Even numberd rows will contain text boxes
        self.textboxes = []
        for i in range(2, self._nrows, 2):
            for j in range(8):
                box = widgets.Text(value='', layout=widgets.Layout(width="90pt"))
                self.layout[i,j] = box
                self.textboxes.append(box)
        
        # Initialize vector
        self.vector = np.zeros(2 ** self.n_qubits, dtype=np.complex64)

    def display(self):
        display(self.layout)

    def update(self, state_vector):
        assert state_vector.size == self.vector.size
        self.vector[:] = state_vector.ravel()
        for i, textbox in enumerate(self.textboxes):
            x = self.vector[i]
            sign = '-' if x.imag < 0 else '+'
            value = f"{x.real:.3f} {sign} {abs(x.imag):.3f}j"
            textbox.value = value

    def _set_callback(self, change):
        # Read values from textboxes
        values = []
        for i, textbox in enumerate(self.textboxes):
            try:
                x = complex(textbox.value.replace(" ", ""))
            except ValueError:
                # Use old value
                x = self.vector[i]
            values.append(x)
        # Update vector
        values = np.asarray(values, dtype=np.complex64)
        values /= np.linalg.norm(values)
        self.update(values)
        # Propagate changes trough mediator
        self.mediator.update("set-components")


class RLAgent:

    def __init__(self, path):
        self.agent = torch.load(path, map_location="cpu")

    def get_policy_and_attentions(self, observation):
        obs = torch.from_numpy(observation)
        nlayers = len(self.agent.policy_network.net)
        net = self.agent.policy_network.net
        scores = []

        with torch.no_grad():
            emb = net[0](obs)
            # First layer is the embedding layer, last two layers are output layers
            for i in range(1, nlayers - 2):
                z_norm = net[i].norm1(emb)
                _, attn = net[i].self_attn(z_norm, z_norm, z_norm,
                    need_weights=True, average_attn_weights=False)
                emb = net[i](emb)
                scores.append(attn.numpy())
            policy = net[-1](net[-2](emb))

        policy = torch.softmax(policy, dim=1)
        policy = np.squeeze(policy.numpy(), 0)
        scores = np.array(scores)
        # Swap (layer, batch, head, X, Y) dimensions to (batch, layer, head, X, Y)
        # Remove `batch` dimension
        scores = scores.swapaxes(0, 1).squeeze(0)
        return policy, scores

    def get_policy(self, observation):
        obs = torch.from_numpy(observation)
        return self.agent.policy(obs).probs[0].numpy()


class InitialStateDropdown:

    def __init__(self, mediator=None, user_defined={}, n_qubits=4):
        self.mediator = mediator
        self.n_qubits = n_qubits
        # Default states
        self._default = get_special_states()[self.n_qubits]
        self._choices = self._default.copy()
        # User defined states
        for key, psi in user_defined.items():
            if psi.size == 2 ** self.n_qubits:
                # Normalize
                self._choices[key] = (psi / np.linalg.norm(psi.ravel())).astype(np.complex64)

        names = list(self._choices.keys())
        self.dropdown = widgets.Dropdown(options=names, index=0,
                                         description=r"$\text{Initial\ state}$")
        self.dropdown.observe(self._callback, "value")
        self._selected = 0

    def _callback(self, change):
        val = change["new"]
        if val != self._selected:
            self._selected = val
            self.mediator.update("reset")
        
    def display(self):
        display(self.dropdown)

    def get_state(self):
        name = self.dropdown.value
        return self._choices[name]


class EntanglementStatus:

    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
        self.status = widgets.HTML(
            value=f"<b style=\"font-size:{BUTTON_FONT_SIZE}\">Single qubit entanglements:</b>")
        self._entanglements = np.zeros(self.n_qubits, dtype=np.float32)

    def dispay(self):
        display(self.status)

    def update(self, entanglements):
        self._entanglements[:] = np.abs(entanglements)
        dtemplate = "<mark style=\"background-color:palegreen;\">{:.3f}</mark>"
        etemplate = "{:.3f}"
        values = []
        for x in self._entanglements:
            if x < 1e-3:
                values.append(dtemplate.format(x))
            else:
                values.append(etemplate.format(x))
        html_value = (4 * "&nbsp").join(values)
        msg = f"<b style=\"font-size:{BUTTON_FONT_SIZE}\">Single qubit entanglements:</b> " + 4 * "&nbsp" + html_value
        self.status.value = msg

    def get_reductions(self):
        return self._entanglements - self._previous


class StepControls:

    def __init__(self, mediator):
        self.mediator = mediator
        self.step_button = widgets.Button(description="Step")
        self.step_button.on_click(self.step)
        self.step_button.style.button_color = COLOR0
        self.step_button.style.font_size = BUTTON_FONT_SIZE

        self.reset_button = widgets.Button(description="Reset")
        self.reset_button.style.button_color = COLOR1
        self.reset_button.style.font_size = BUTTON_FONT_SIZE
        self.reset_button.on_click(self.reset)

        self.undo_button = widgets.Button(description="Undo", disabled=True)
        self.undo_button.style.button_color = COLOR1
        self.undo_button.style.font_size = BUTTON_FONT_SIZE
        self.undo_button.on_click(self.undo)

        self.redo_button = widgets.Button(description="Redo", disabled=True)
        self.redo_button.style.button_color = COLOR1
        self.redo_button.style.font_size = BUTTON_FONT_SIZE
        self.redo_button.on_click(self.redo)

        self.layout = widgets.GridspecLayout(1, 4)
        self.layout[0,0] = self.reset_button
        self.layout[0,1] = self.step_button
        self.layout[0,2] = self.undo_button
        self.layout[0,3] = self.redo_button

    def display(self):
        display(self.layout)

    def step(self, button):
        self.mediator.update("step")

    def reset(self, button):
        self.mediator.update("reset")

    def undo(self, button):
        self.mediator.update("undo")

    def redo(self, button):
        self.mediator.update("redo")


class History:

    def __init__(self):
        self._checkpoints = []
        self._pointer = 0

    def save(self, states, rot_ctrl_1q, rot_ctrl_2q, attn, policy):
        self.edits()
        self._checkpoints.append((states, rot_ctrl_1q, rot_ctrl_2q, attn, policy))
        self._pointer += 1

    def empty(self):
        return not self._checkpoints
    
    def reset(self):
        self._checkpoints.clear()
        self._pointer = 0

    def undo(self, states, rot_ctrl_1q, rot_ctrl_2q, attn, policy):
        if self.can_undo():
            checkpoint = self._checkpoints[self._pointer - 1]
            # Add current state to history or overwrite
            if self._pointer == len(self._checkpoints):
                self._checkpoints.append(
                    (states, rot_ctrl_1q, rot_ctrl_2q, attn, policy))
            else:
                self._checkpoints[self._pointer] = \
                    (states, rot_ctrl_1q, rot_ctrl_2q, attn, policy)
            self._pointer -= 1
            return checkpoint
        return None

    def redo(self):
        if self.can_redo():
            self._pointer += 1
            checkpoint = self._checkpoints[self._pointer]
            return checkpoint
        return None

    def edits(self):
        # Clear all states after pointer
        while self._pointer < len(self._checkpoints):
            self._checkpoints.pop()

    def can_undo(self):
        return self._pointer > 0

    def can_redo(self):
        return self._pointer + 1 < len(self._checkpoints)


class DummyFigure:

    def update(self, *args, **kwargs):
        pass

    def get_scores(self):
        pass

    def get_policy(self):
        pass


class CircuitFigure:

    config = {
        "circle-fontsize": 12,
        "circle-radius": 0.4,
        "circle-linewidth": 1,
        "wire-linewidth": 0.25,
        "gate-linewidth": 2.0,
        "gate-marker-size": 8.0,
        "step-size": 1,
        "default-xlim-4q": 12,
        "default-xlim-5q": 24,
    }

    def __init__(self, ax, n_qubits=4):
        self.n_qubits = n_qubits
        self.step = 0
        if self.n_qubits == 4:
            self.default_xlim = CircuitFigure.config["default-xlim-4q"]
        else:
            self.default_xlim = CircuitFigure.config["default-xlim-5q"]

        # Containers for Artist objects
        self.gates, self.wires = [], []
        # Initialize ax
        self.ax = ax
        self.reset_ax()
        self.draw_circles()
        self.draw_wires()

    def reset(self):
        # Remove gates
        for i in range(self.step):
            self.gates[i].remove()
        self.step = 0
        self.gates.clear()
        # Remove wires
        for wire in self.wires:
            wire.remove()
        self.wires.clear()
        # Redraw wires
        self.draw_wires()
        # Reset step and ax
        self.reset_ax()

    def reset_ax(self):
        self.ax.set_aspect(1.0)
        self.ax.set_xlim(-1, self.default_xlim)
        self.ax.set_ylim(-1, self.n_qubits)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_visible(False)

    def draw_circles(self):
        N = self.n_qubits
        qubits_ys = np.arange(0, N)
        qubits_xs = np.full(N, 0)
        r = CircuitFigure.config["circle-radius"]
        styledict = dict(edgecolor='k', fill=True, facecolor="white", zorder=10,
            linewidth=CircuitFigure.config["circle-linewidth"],
        )
        textdict = dict(fontsize=CircuitFigure.config["circle-fontsize"],
            ha="center", va="center", zorder=11
        )
        for x, y in zip(qubits_xs, qubits_ys):
            self.ax.add_patch(patches.Circle((x, y), r, **styledict))
            self.ax.text(x, y, f'$q_{y+1}$', fontdict=textdict)

    def draw_wires(self):
        step_size = CircuitFigure.config["step-size"]
        wmin_x = CircuitFigure.config["circle-radius"]
        wmax_x = max(self.ax.get_xlim()[1] + 1, len(self.gates) + 4 * step_size)
        styledict = dict(zorder=0, color='k', alpha=0.5,
            linewidth=CircuitFigure.config["wire-linewidth"]
        )
        for i in range(self.n_qubits):
            l = lines.Line2D([wmin_x, wmax_x], [i, i], **styledict)
            self.wires.append(l)
            self.ax.add_line(l)

    def relim_xaxis(self):
        xlim = self.ax.get_xlim()[1]
        step_size = CircuitFigure.config["step-size"]
        wire_redraw = False
        if (self.step + 1) * step_size >= xlim:
            self.ax.set_xlim(-1, (self.step + 2) * step_size)
            wire_redraw = True
        elif xlim == self.default_xlim:
            return
        elif xlim - (self.step * step_size) >= 4 * step_size:
            xlim = max(self.default_xlim, (self.step + 4) * step_size)
            self.ax.set_xlim(-1, xlim)
            wire_redraw = True
        # Redraw wires
        if wire_redraw:
            for wire in self.wires:
                wire.remove()
            self.wires.clear()
            self.draw_wires()

    def append(self, action):
        # Clear everything after `self.step`
        while len(self.gates) > self.step:
            self.gates.pop()
        self.step += 1
        step_size = CircuitFigure.config["step-size"]
        x = self.step * step_size

        q0, q1 = sorted(action)
        gate = lines.Line2D(
            [x, x],
            [q0, q1],
            linewidth=CircuitFigure.config["gate-linewidth"],
            color='k',
            marker='o',
            markersize=CircuitFigure.config["gate-marker-size"]
        )
        self.gates.append(gate)
        self.ax.add_line(gate)
        self.relim_xaxis()
        self.ax.set_xticks(np.arange(1, self.step+1))

    def undo(self):
        if self.step == 0:
            return
        self.step -= 1
        self.gates[self.step].remove()
        self.relim_xaxis()
        self.ax.set_xticks(np.arange(1, self.step+1))

    def redo(self):
        if self.step == len(self.gates):
            return
        self.ax.add_line(self.gates[self.step])
        self.step += 1
        self.relim_xaxis()
        self.ax.set_xticks(np.arange(1, self.step+1))


class PolicyFigure:

    def __init__(self, ax, n_qubits=4):
        self.n_qubits = n_qubits

        if ax is None:
            fig = plt.figure(frameon=False, layout="none", figsize=(5, 2))
            ax = fig.add_axes((0.2, 0.2, 0.7, 0.8))
            self.figure = fig
            self.ax = ax
        else:
            self.ax = ax
            self.figure = None
        
        N = (n_qubits * (n_qubits - 1)) // 2
        self._policy     = np.zeros(N, dtype=np.float32)
        self._reductions = np.zeros(N, dtype=np.float32)
        a = itertools.combinations(range(1, n_qubits+1), 2)
        self._actions = [(i,j) for i,j in a if i < j]
        self._xs = np.arange(N)
        self._bars  = self.ax.bar(self._xs, self._policy,
                                 color="tab:blue", width=0.9)
        self._nbars = self.ax.bar(self._xs, self._reductions,
                                  color="lightgreen", width=0.9)
        self._percentages = []
        self._reductions = []
        for x in self._xs:
            self._percentages.append(self.ax.text(x, 0.2, s='', ha="center"))
            self._reductions.append(self.ax.text(x, -0.2, s='', ha="center"))

        self.ax.set_xticks(self._xs, self._actions)
        self.ax.set_yticks([])
        self.ax.set_ylim(-0.9, 1.2)
        self.ax.set_xlim(-1.5, N+0.5)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        # Ensure tick labels are still visible
        self.ax.xaxis.set_tick_params(which='both', labelbottom=True)
        # Add text labels
        self.ax.text(x=-1.0, y=0.3,
                     s=r"$\pi(a|o)$", fontsize=12, ha="center")
        self.ax.text(x=-1.0, y=-0.3,
                     s=r"$\Delta S_\mathrm{avg}$", fontsize=12, ha="center")
        self.ax.text(x=-1.0, y=-1.0,
                     s=r"$\mathrm{action}$", fontsize=12, ha="center")
        self.ax.axhline(0.0, linewidth=1.0, color='k')
    
    def update(self, policy, entanglement_reductions=None):
        self._policy[:] = np.asarray(policy).ravel()
        a = np.argmax(policy)
        percentages = (100 * self._policy).astype(np.int32)
        # Compensate for rounding errors
        percentages[a] += (100 - np.sum(percentages))
        # Change bar heights and % annotation & selected action color
        for i in range(len(self._bars)):
            self._bars[i].set_height(self._policy[i])
            self._percentages[i].set_y(self._policy[i] + 0.05)
            self._percentages[i].set_text(f"${percentages[i]}\\%$")
            if entanglement_reductions is not None:
                val = min(0.0, -entanglement_reductions[i])
                self._nbars[i].set_height(val)
                self._reductions[i].set_y(val - 0.2)
                self._reductions[i].set_text(f"${val:.2f}$")
            if i == a:
                self._bars[i].set_color("tab:red")
            else:
                self._bars[i].set_color("tab:blue")

    def reset(self):
        self._policy[:] = 0.0
        for i in range(len(self._actions)):
            self._bars[i].set_height(0.0)
            self._nbars[i].set_height(0.0)
            self._bars[i].set_color("tab:blue")
            self._percentages[i].set_text('')
            self._reductions[i].set_text('')

    def get_policy(self):
        return self._policy.copy()


class AttentionsFigure:

    def __init__(self, axs, n_layers, n_heads, n_qubits=4):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.axs = np.asarray(axs)

        # Set image data
        self._images = []
        self._actions = get_action_pairs(self.n_qubits)
        N = len(self._actions)
        self._attention_scores = np.zeros((n_layers, n_heads, N, N),
                                          dtype=np.float32)
        for ax in self.axs.flat:
            im = ax.imshow(self._attention_scores[0][0],
                           cmap="gray", vmin=0.0, vmax=1.0)
            self._images.append(im)

    def update(self, attention_scores):
        assert self.n_layers == attention_scores.shape[0]
        assert self.n_heads == attention_scores.shape[1]
        self._attention_scores[:] = attention_scores

        imiter = iter(self._images)
        for i,j in itertools.product(range(self.n_layers), range(self.n_heads)):
            im = next(imiter)
            im.set_data(attention_scores[i][j])

    def reset(self):
        self._attention_scores[:] = 0.0
        self.update(self._attention_scores)

    def get_scores(self):
        return self._attention_scores.copy()


class DummyMediator:

    def update(self, message, *args, **kwargs):
        print(f"Call to DummyMediator.update(\"{message}\", {args}, {kwargs}).")


class Mediator:

    def __init__(
            self,
            simulator,              # Quantum state simulator
            agent,                  # Wrapper of RL agent
            # UI Elements
            reset_dropdown,         # Dropdown menu with start states
            amplitudes_box,         # Quantum state components box
            entanglements_box,      # Entanglements box
            rot_1q_ctrl,            # Single-qubit rotation control box
            rot_2q_ctrl,            # Two-qubit rotation control box
            step_ctrl,              # Step/Undo control box
            circuit_fig,            # Figure of quantum cirquit
            attentions_fig,         # Figure of attention scores
            policy_fig,             # Figure of action probabilities
            history                 # History
    ):
        self._reset_dropdown = reset_dropdown
        self._amplitudes_box = amplitudes_box
        self._simulator = simulator
        self._agent = agent
        self._entanglements_box = entanglements_box
        self._rotation_ctrl_1q = rot_1q_ctrl
        self._rotation_ctrl_2q = rot_2q_ctrl
        self._step_ctrl = step_ctrl
        self._circuit_fig = circuit_fig
        self._attentions_fig = attentions_fig
        self._policy_fig = policy_fig
        self._history = history

    def update(self, message, *args, **kwargs):

        match message:

            case "rotate-1q":
                # Apply rotation
                psi = self._simulator.states[0]
                phi = apply_1q_rotation(psi, *args, **kwargs)
                # Update amplitudes box
                self._amplitudes_box.update(phi.ravel())
                # Update simulator's state
                self._simulator.states = phi.reshape(self._simulator.states.shape)
                # Update policy figure
                obs = observe(self._simulator.states)
                policy, attn = self._agent.get_policy_and_attentions(obs)
                ent_reduction = calc_entanglement_reduction(phi)
                self._policy_fig.update(policy, ent_reduction)
                # Update attentions figure
                self._attentions_fig.update(attn)
                # TODO Maybe show rotation gate on `circuit`?
                # Disable Redo button
                self._history.edits()
                self._step_ctrl.redo_button.disabled = True
        
            case "rotate-2q":
                psi = self._simulator.states[0]
                phi = apply_2q_rotation(psi, *args, **kwargs)
                # Update amplitudes box
                self._amplitudes_box.update(phi.ravel())
                # Update simulator's state
                self._simulator.states = phi.reshape(self._simulator.states.shape)
                # Update policy figure
                obs = observe(self._simulator.states)
                entred = calc_entanglement_reduction(phi)
                policy, attn = self._agent.get_policy_and_attentions(obs)
                self._policy_fig.update(policy, entred)
                # Update attentions figure
                self._attentions_fig.update(attn)
                # Update entanglements status
                ent = calc_entanglement(self._simulator.states)
                self._entanglements_box.update(ent[0])
                # TODO Maybe show rotation gate on `circuit`?
                # Disable Redo button
                self._history.edits()
                self._step_ctrl.redo_button.disabled = True

            case "step":
                # Save the current state of almost everything
                self._history.save(self._simulator.states.copy(),
                                   self._rotation_ctrl_1q.get_state(),
                                   self._rotation_ctrl_2q.get_state(),
                                   self._attentions_fig.get_scores(),
                                   self._policy_fig.get_policy())
                # Get action from the RL agent
                obs = observe(self._simulator.states)
                policy = self._agent.get_policy(obs)
                action = np.argmax(policy)
                # Update the RL environment state
                self._simulator.apply([action])
                # Update the Quantum State components
                self._amplitudes_box.update(self._simulator.states[0].ravel())
                # Reset rotation controls
                self._rotation_ctrl_1q.reset()
                self._rotation_ctrl_2q.reset()
                # Update attentions figure
                obs = observe(self._simulator.states)
                policy, attn = self._agent.get_policy_and_attentions(obs)
                self._attentions_fig.update(attn)
                # Update circuit figure
                gate = self._simulator.actions[action]
                self._circuit_fig.append(gate)
                # Update policy figure
                entred = calc_entanglement_reduction(self._simulator.states)
                self._policy_fig.update(policy, entred)
                # Update entanglements box
                ent = calc_entanglement(self._simulator.states)
                self._entanglements_box.update(ent[0])
                # Enable the Undo button
                if self._history.can_undo():
                    self._step_ctrl.undo_button.disabled = False
                # Disable the Redo button
                self._step_ctrl.redo_button.disabled = True

            case "reset":
                # Get initial state from dropdown menu
                state = self._reset_dropdown.get_state()
                # Reset environment
                self._simulator.states = state.reshape(self._simulator.states.shape)
                # Reset rotation controls
                self._rotation_ctrl_1q.reset()
                self._rotation_ctrl_2q.reset()
                # Reset circuit figure
                self._circuit_fig.reset()
                # Update amplitudes box
                self._amplitudes_box.update(state)
                # Update entanglements box
                ent = calc_entanglement(self._simulator.states)
                self._entanglements_box.update(ent[0])
                # Update attention figure
                obs = observe(self._simulator.states)
                policy, attn = self._agent.get_policy_and_attentions(obs)
                self._attentions_fig.update(attn)
                # Update policy figure
                entred = calc_entanglement_reduction(self._simulator.states)
                self._policy_fig.update(policy, entred)
                # Reset history
                self._history.reset()
                # Disable Undo & Redo buttons
                self._step_ctrl.undo_button.disabled = True
                self._step_ctrl.redo_button.disabled = True

            case "undo":
                if not self._history.can_undo():
                    return
                psi, rot1q, rot2q, attn, policy = self._history.undo(
                    self._simulator.states.copy(),
                    self._rotation_ctrl_1q.get_state(),
                    self._rotation_ctrl_2q.get_state(),
                    self._attentions_fig.get_scores(),
                    self._policy_fig.get_policy()
                )
                # Update simulator's state
                self._simulator.states = psi
                # Set rotation controls
                self._rotation_ctrl_1q.set_state(rot1q)
                self._rotation_ctrl_2q.set_state(rot2q)
                # Update quantum state components box
                self._amplitudes_box.update(psi.ravel())
                # Update circuit figure
                self._circuit_fig.undo()
                # Update entanglements box
                ent = calc_entanglement(self._simulator.states)
                self._entanglements_box.update(ent[0])
                # Update attention figure
                self._attentions_fig.update(attn)
                # Update policy figure
                entred = calc_entanglement_reduction(self._simulator.states)
                self._policy_fig.update(policy, entred)
                # Disable the "Undo" button
                if not self._history.can_undo():
                    self._step_ctrl.undo_button.disabled = True
                # Enable the "Redo" button
                if self._history.can_redo():
                    self._step_ctrl.redo_button.disabled = False

            case "redo":
                if not self._history.can_redo():
                    return
                psi, rot1q, rot2q, attn, policy = self._history.redo()
                # Update simulator's state
                self._simulator.states = psi
                # Set rotation controls
                self._rotation_ctrl_1q.set_state(rot1q)
                self._rotation_ctrl_2q.set_state(rot2q)
                # Update quantum state components box
                self._amplitudes_box.update(psi.ravel())
                # Update circuit figure
                self._circuit_fig.redo()
                # Update entanglements box
                ent = calc_entanglement(self._simulator.states)
                self._entanglements_box.update(ent[0])
                # Update attention figure
                self._attentions_fig.update(attn)
                # Update policy figure
                entred = calc_entanglement_reduction(self._simulator.states)
                self._policy_fig.update(policy, entred)
                # Disable the "Redo" button
                if not self._history.can_redo():
                    self._step_ctrl.redo_button.disabled = True
                # Enable the "Undo" button
                if self._history.can_undo():
                    self._step_ctrl.undo_button.disabled = False

            case "set-components":
                # Get the quantum state vector from components box
                vector = self._amplitudes_box.vector
                # Update RL environment state
                self._simulator.states = vector.reshape(self._simulator.states.shape)
                # Update entanglements box
                ent = calc_entanglement(self._simulator.states)
                self._entanglements_box.update(ent[0])
                # Update policy figure
                obs = observe(self._simulator.states)
                policy, attn = self._agent.get_policy_and_attentions(obs)
                entred = calc_entanglement_reduction(self._simulator.states)
                self._policy_fig.update(policy, entred)
                # Update attention figure
                self._attentions_fig.update(attn)
                # Disable Redo button
                self._history.edits()
                self._step_ctrl.redo_button.disabled = True



def get_action_pairs(num_qubits):
    x = itertools.combinations(range(1, num_qubits + 1), 2)
    return [(i,j) for i,j in x if i < j]


def observe_widgets(_widgets, _handlers):
    for w, h in zip(_widgets, _handlers):
        w.observe(h, "value")


def unobserve_widgets(_widgets, _handlers):
    for w, h in zip(_widgets, _handlers):
        w.unobserve(h, "value")


def get_special_states():
    """Returns dictionaty with special states for 4 & 5 qubits."""

    bell = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)]).astype(np.complex64)
    x = 1 / np.sqrt(3)
    w = np.array([0, x, x, 0, x, 0, 0, 0], dtype=np.complex64).reshape(2,2,2)
    ghz = np.zeros(8, dtype=np.complex64)
    ghz[0] = 1 / np.sqrt(2)
    ghz[7] = 1 / np.sqrt(2)
    ghz = ghz.reshape(2,2,2).astype(np.complex64)
    zero = np.array([1, 0]).astype(np.complex64)
    bell = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex64) / np.sqrt(2)
    bell = bell.reshape(2,2)
    zero = np.array([1, 0], dtype=np.complex64)
    haar1 = random_quantum_state(q=1, prob=1.0)
    haar2 = random_quantum_state(q=2, prob=1.0).reshape(2,2)
    haar3 = random_quantum_state(q=3, prob=1.0).reshape(2,2,2)
    haar4 = random_quantum_state(q=4, prob=1.0).reshape(2,2,2,2)
    haar5 = random_quantum_state(q=5, prob=1.0).reshape(2,2,2,2,2)

    special_states = {
        4: {
            "|0>|Bell>|0>": np.einsum("i,jk,l->ijkl", zero, bell, zero),
            "|Bell>|R>|R>": np.einsum("ij,k,l->ijkl", bell, haar1, haar1),
            "|W>|R>"      : np.einsum("ijk,l->ijkl", w, haar1),
            "|GHZ>|R>"    : np.einsum("ijk,l->ijkl", ghz, haar1),
            "|RR>|R>|R>"  : np.einsum("ij,k,l->ijkl", haar2, haar1, haar1),
            "|RR>|RR>"    : np.einsum("ij,kl->ijkl", haar2, haar2),
            "|RRR>|R>"    : np.einsum("ijk,l->ijkl", haar3, haar1),
            "|RRRR>"      : haar4,
        },
        5: {
            "|0>|Bell>|0>|0>": np.einsum("i,jk,l,m->ijklm", zero, bell, zero, zero),
            "|Bell>|Bell>|0>": np.einsum("ij,kl,m->ijklm", bell, bell, zero),
            "|W>|0>|0>"      : np.einsum("ijk,l,m->ijklm", w, zero, zero),
            "|W>|Bell>"      : np.einsum("ijk,lm->ijklm", w, bell),
            "|W>|RR>"        : np.einsum("ijk,lm->ijklm", ghz, haar2),
            "|GHZ>|Bell>"    : np.einsum("ijk,lm->ijklm", ghz, bell),
            "|GHZ>|R>|R>"    : np.einsum("ijk,l,m->ijklm", ghz, haar1, haar1),
            "|GHZ>|RR>"      : np.einsum("ijk,lm->ijklm", ghz, haar2),
            "|RR>|R>|R>|R>"  : np.einsum("ij,k,l,m->ijklm", haar2, haar1, haar1, haar1),
            "|RR>|RR>|R>"    : np.einsum("ij,kl,m->ijklm", haar2, haar2, haar1),
            "|RRR>|R>|R>"    : np.einsum("ijk,l,m->ijklm", haar3, haar1, haar1),
            "|RRR>|RR>"      : np.einsum("ijk,lm->ijklm", haar3, haar2),
            "|RRRR>|R>"      : np.einsum("ijkl,m->ijklm", haar4, haar1),
            "|RRRRR>"        : haar5
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


def apply_1q_rotation(psi, qubit, phi, theta, angle):
    # Formula
    # |\psi_\text{new}\rangle = e^{-i(\eta/2)[\sin(\theta)\cos(\phi)X_i + \sin(\theta)\sin(\phi)Y_i + \cos(\theta)Z_i]}|\psi\rangle
    h = [np.sin(theta) * np.cos(phi),
         np.sin(theta) * np.sin(phi),
         np.cos(theta)]
    M = h[0] * SIGMA_X + h[1] * SIGMA_Y + h[2] * SIGMA_Z
    U = expm(-1.0j * (angle / 2) * M)
    n_qubits = int(np.log2(psi.size))
    U = unitary_for_nqubits(U, n_qubits, qubit)
    return np.matmul(U, psi.reshape(-1, 1)).reshape(psi.shape)


def apply_2q_rotation(psi, qubit0, qubit1, alpha, beta, gamma):
    # Formula
    # |\psi_\text{new}\rangle = e^{-i[\alpha X_iX_j + \beta Y_iY_j + \gamma Z_iZ_j]}|\psi\rangle
    n_qubits = int(np.log2(psi.size))
    original_shape = psi.shape
    shape = (2,) * n_qubits
    psi = psi.reshape(shape)
    qubits = [qubit0, qubit1]
    T = qubits + [k for k in range(n_qubits) if k not in qubits]
    psi = np.transpose(psi, T)
    sigma_x = np.kron(SIGMA_X, SIGMA_X)
    sigma_y = np.kron(SIGMA_Y, SIGMA_Y)
    sigma_z = np.kron(SIGMA_Z, SIGMA_Z)
    M = alpha * sigma_x + beta * sigma_y + gamma * sigma_z
    U = expm(-1.0j * M)
    psi = np.matmul(U, psi.reshape(4, -1)).reshape(shape)
    psi = np.transpose(psi, np.argsort(T))
    return psi.reshape(original_shape)


def calc_entanglement_reduction(state):
    n_qubits = int(np.log2(state.size))
    n_actions = ((n_qubits) * (n_qubits - 1)) // 2
    simulator = QStateSimulator(n_qubits, num_envs=n_actions, act_space="reduced")
    state = state.reshape((1,) + (2,) * n_qubits)
    states = np.tile(state, (n_actions,) + (1,) * n_qubits)
    simulator.states = states

    current_entanglements = simulator.entanglements.copy()
    actions = np.arange(n_actions)
    simulator.apply(actions)
    next_entanglements = simulator.entanglements.copy()
    reductions = current_entanglements - next_entanglements
    return np.mean(reductions, axis=1)


def start_demo_4q(user_defined_states):

    # Initialize figure
    with plt.ioff():
        fig = plt.figure(figsize=(11, 5), layout="none")
    # Always hide the toolbar
    fig.canvas.toolbar_visible = False
    # Hide the Figure name at the top of the figure
    fig.canvas.header_visible = False
    # Hide the footer
    fig.canvas.footer_visible = False
    # Disable the resizing feature
    fig.canvas.resizable = False

    gridspec = fig.add_gridspec(nrows=20, ncols=44, left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax_circuit = fig.add_subplot(gridspec[1:8, :20])
    ax_policy = fig.add_subplot(gridspec[11:19, :20])

    ax_attn11 = fig.add_subplot(gridspec[2:8, 26:32])
    ax_attn12 = fig.add_subplot(gridspec[2:8, 33:39])
    ax_attn21 = fig.add_subplot(gridspec[9:15, 26:32])
    ax_attn22 = fig.add_subplot(gridspec[9:15, 33:39])

    ax_colorbar = fig.add_subplot(gridspec[2:15, 40:41])

    # ax_circuit.set_title("Circuit")
    ax_circuit.set_xlabel("Episode Step", fontsize=12)
    ax_attn11.set_title("Layer 1, Head 1", fontsize=10)
    ax_attn12.set_title("Layer 1, Head 2", fontsize=10)
    ax_attn21.set_title("Layer 2, Head 1", fontsize=10)
    ax_attn22.set_title("Layer 2, Head 2", fontsize=10)
    actions = get_action_pairs(4)
    labels = [f"$x^{{{a}}}$" for a in actions]
    ax_attn11.set_xticks([])
    ax_attn11.set_yticks(np.arange(len(labels)), labels)
    ax_attn12.set_xticks([])
    ax_attn12.set_yticks([])
    ax_attn21.set_yticks(np.arange(len(labels)), labels)
    ax_attn21.set_xticks(np.arange(len(labels)), labels, rotation=45)
    ax_attn22.set_yticks([])
    ax_attn22.set_xticks(np.arange(len(labels)), labels, rotation=45)
    ax_attn11.text(x=4.0, y=-2.0, s="Attention Scores", fontsize=12) 

    cmap = mpl.cm.gray
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_colorbar)
    # Initialize supporting objects
    simulator = QStateSimulator(4, 1)
    agent = RLAgent(PATH_4Q_AGENT)

    # Initialize widgets
    init_state_dropdown = InitialStateDropdown(None, user_defined_states, 4)
    rot1q = RotationControls1q(None, 4)
    rot2q = RotationControls2q(None, 4)
    step_ctrls = StepControls(None)
    ent_status = EntanglementStatus(4)
    amplitudes = AmplitudesBox(None, 4)
    attn_fig = AttentionsFigure([ax_attn11, ax_attn12, ax_attn21, ax_attn22], 2, 2, 4)

    policy_fig = PolicyFigure(ax_policy, 4)
    circuit_fig = CircuitFigure(ax_circuit, 4)
    history = History()

    # Initialize mediator
    mediator = Mediator(simulator, agent, init_state_dropdown, amplitudes,
                        ent_status, rot1q, rot2q, step_ctrls, circuit_fig,
                        attn_fig, policy_fig, history)

    # Update mediator attributes
    rot1q.mediator = mediator
    rot2q.mediator = mediator
    step_ctrls.mediator = mediator
    amplitudes.mediator = mediator
    init_state_dropdown.mediator = mediator

    # Single Qubit Rotation
    rot1q.display()
    # Two Qubit Rotation
    rot2q.display()
    # Reset / Step / Undo / Redo controls
    controls_layout = widgets.GridspecLayout(1, 6)
    controls_layout[0, 0:1] = init_state_dropdown.dropdown
    controls_layout[0, 1:2] = step_ctrls.reset_button
    controls_layout[0, 3:4] = step_ctrls.step_button
    controls_layout[0, 4:5] = step_ctrls.undo_button
    # Disabled, because it currently acts as a duplicate of Step button
    # controls_layout[0, 5:6] = step_ctrls.redo_button
    display(controls_layout)
    # Single qubit entanglements
    ent_status.dispay()
    display(fig.canvas)
    # Add empty row
    display(widgets.HTML(value="<br>"))

    # Amplitudes
    amplitudes.display()

    mediator.update("reset")


def start_demo_5q(user_defined_states):

    # Initialize figure
    with plt.ioff():
        fig = plt.figure(figsize=(11, 6), layout="none")
    # Always hide the toolbar
    fig.canvas.toolbar_visible = False
    # Hide the Figure name at the top of the figure
    fig.canvas.header_visible = False
    # Hide the footer
    fig.canvas.footer_visible = False
    # Disable the resizing feature
    fig.canvas.resizable = False
    gridspec = fig.add_gridspec(nrows=24, ncols=44, left=0.01, right=0.99, top=0.99, bottom=0.01)
    ax_circuit = fig.add_subplot(gridspec[:11, :])
    ax_policy = fig.add_subplot(gridspec[13:23, :])
    ax_circuit.set_xlabel("Episode Step", fontsize=12)

    # Initialize supporting objects
    simulator = QStateSimulator(5, 1)
    agent = RLAgent(PATH_5Q_AGENT)

    # Initialize widgets
    init_state_dropdown = InitialStateDropdown(None, user_defined_states, 5)
    rot1q = RotationControls1q(None, 5)
    rot2q = RotationControls2q(None, 5)
    step_ctrls = StepControls(None)
    ent_status = EntanglementStatus(5)
    amplitudes = AmplitudesBox(None, 5)

    policy_fig = PolicyFigure(ax_policy, 5)
    circuit_fig = CircuitFigure(ax_circuit, 5)
    history = History()
    attn_fig = DummyFigure()

    # Initialize mediator
    mediator = Mediator(simulator, agent, init_state_dropdown, amplitudes,
                        ent_status, rot1q, rot2q, step_ctrls, circuit_fig,
                        attn_fig, policy_fig, history)

    # Update mediator attributes
    rot1q.mediator = mediator
    rot2q.mediator = mediator
    step_ctrls.mediator = mediator
    amplitudes.mediator = mediator
    init_state_dropdown.mediator = mediator

    # Single Qubit Rotation
    rot1q.display()
    # Two Qubit Rotation
    rot2q.display()
    # Reset / Step / Undo / Redo controls
    controls_layout = widgets.GridspecLayout(1, 6)
    controls_layout[0, 0:1] = init_state_dropdown.dropdown
    controls_layout[0, 1:2] = step_ctrls.reset_button
    controls_layout[0, 3:4] = step_ctrls.step_button
    controls_layout[0, 4:5] = step_ctrls.undo_button
    # Disabled, because it currently acts as a duplicate of Step button
    # controls_layout[0, 5:6] = step_ctrls.redo_button
    display(controls_layout)
    # Single qubit entanglements
    ent_status.dispay()
    # Figure
    display(fig.canvas)
    # Add empty row
    display(widgets.HTML(value="<br>"))
    # Amplitudes
    amplitudes.display()

    mediator.update("reset")
