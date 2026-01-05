import copy
import dataclasses
import logging
import os
from typing import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import metrics
from . import stategen
from .config import get_logdir
from .evaluation import test_lengths
from .qenv import QEnv


@dataclasses.dataclass
class StagedTrainingLevel:
    """
    Level description for `StagedTrainingTrigger`.

    A level specifies the test parameters with which the condition is tested
    each time `StagedTrainingTrigger` is run, the condition thresholds and the
    parameters that are updated for the RL environment, RL agent and state
    generation object.
    """
    # Test specific parameters. In order to check if the level condition is
    # fulfilled, the RL agent is tested with a (possibly different from the
    # training one) StateGenerator object.
    test_max_steps: int                 # Maximum rollout steps in the test
    test_min_subsystem_size: int        # Minimum subsystem size
    test_max_subsystem_size: int        # Maximum subsystem size

    threshold_average_length: float
    threshold_ratio_terminated: float

    env_parameters: dict
    sgen_parameters: dict
    agent_parameters: dict

    # Minimum number of iterations to be spent on current level, before
    # transitioning to next one
    min_iterations: int = 1
    test_min_eta: float = 10.0
    test_max_eta: float = 10.0

class StagedTrainingTrigger:

    def __init__(self, config, agent, env):
        self.config = config
        self.agent = agent
        self.env = env
        self.levels = self._parse_levels(config)
        self.current_level = 0
        self.final_level = len(self.levels) - 1

    def __call__(self, *args, **kwargs):

        tracker = metrics.getTracker()
        tracker.add_scalar("[StagedTrainingTrigger] Level", self.current_level)

        # Check if we're at the final level and return immediately if so
        if self.current_level == self.final_level:
            return

        # Check if levels ...
        if not self.levels:
            return

        logging.info(f"\n[StagedTrainingTrigger] Level {self.current_level}")

        if self.condition():
            self.action()
        else:
            logging.info("\n\tCondition not satisfied. "
                         f"Staying on level {self.current_level}")

    def condition(self):
        # Get current level parameters
        level = self.levels[self.current_level]
        level.min_iterations -= self.config["trigger_every"]

        # Initialize state sampler to test agent accuracy on the current level
        test_sampler = copy.deepcopy(self.env.state_generator)
        test_sampler.sample_params.update(
            dict(
                min_subsystem_size=level.test_min_subsystem_size,
                max_subsystem_size=level.test_max_subsystem_size,
            )
        )
        if "min_eta" in test_sampler.sample_params:
            test_sampler.sample_params.update(dict(min_eta=level.test_min_eta))
        if "max_eta" in test_sampler.sample_params:
            test_sampler.sample_params.update(dict(max_eta=level.test_max_eta))

        logging.info("\tInitialized state sampler with parameters:")
        for k, v in test_sampler.sample_params.items():
            logging.info(f"\t\t{k} = {v}")

        # Test the agent
        logging.info(f"\tTesting agent with maximum rollout steps = {level.test_max_steps}")
        lengths = test_lengths(
            self.agent,
            test_sampler,
            level.test_max_steps,
            obs_fn=self.config.obs_fn,
            num_tests=self.config.num_tests,
            epsi=self.config.epsi,
            greedy=self.config.greedy_evaluation_policy
        )

        # Calculate ratio of disentangled states and average solution length
        ratio_terminated = np.sum(~np.isnan(lengths)) / lengths.size
        avg_length = np.nanmean(lengths)
        std_length = np.nanstd(lengths)
        min_length = np.nanmin(lengths)
        max_length = np.nanmax(lengths)
        p90_length = np.nanpercentile(lengths, 90)
        if np.isnan(avg_length):
            avg_length = np.inf
            std_length = np.inf
            min_length = np.inf
            max_length = np.inf
            p90_length = np.inf
        logging.info(f"\tRatio terminated: {ratio_terminated:.3f} "
                     f"({level.threshold_ratio_terminated:.2f})")
        logging.info(f"\tAverage length:   {avg_length:.1f} ± {std_length:.1f} "
                     f"({level.threshold_average_length:.2f})")
        logging.info(f"\tMinimum length:   {min_length:.1f}")
        logging.info(f"\tMaximum length:   {max_length:.1f}")
        logging.info(f"\t90-th % length:   {p90_length:.1f}")

        tracker = metrics.getTracker()
        tracker.add_scalar("[StagedTrainingTrigger] Max Subsystem Size", level.test_max_subsystem_size)
        tracker.add_scalar("[StagedTrainingTrigger] Ratio Terminated", ratio_terminated)
        tracker.add_scalar("[StagedTrainingTrigger] Average Length", avg_length)

        # Test condition
        if (avg_length <= level.threshold_average_length and
            ratio_terminated >= level.threshold_ratio_terminated and
            level.min_iterations <= 0):
            return True
        else:
            return False

    def action(self):

        # Transition
        self.current_level += 1
        logging.info(f"\n\tTransitioning to level {self.current_level}...\n")

        # Get new level parameters
        level = self.levels[self.current_level]

        # Update state generator & log
        for name, val in level.sgen_parameters.items():
            setattr(self.env.state_generator, name, val)
            newval = getattr(self.env.state_generator, name)
            logging.info(f"\t\t[State Generator] {name} = {newval}")

        # Update and reset RL environment & log
        for name, val in level.env_parameters.items():
            setattr(self.env, name, val)
            newval = getattr(self.env, name)
            logging.info(f"\t\t[Environment] {name} = {newval}")
        self.env.reset()

        # Update agent parameters & log
        for name, val in level.agent_parameters.items():
            setattr(self.agent, name, val)
            newval = getattr(self.agent, name)
            logging.info(f"\t\t[Agent] {name} = {newval}")

    def state_dict(self):
        return dict(
            type =              str(type(self)),
            current_level =     self.current_level,
            final_level =       self.final_level,
            levels =            [dataclasses.asdict(l) for l in self.levels]
        )

    def load_state_dict(self, state_dict):

        assert self.final_level == state_dict["final_level"]
        assert len(self.levels) == len(state_dict["levels"])

        for l_self, l_other in zip(self.levels, state_dict["levels"]):
            assert dataclasses.asdict(l_self) == l_other

        self.current_level = state_dict["current_level"]
        if self.current_level > 0:
            self.current_level -= 1
            self.action()

    def _parse_levels(self, config) -> List[StagedTrainingLevel]:
        levels = []
        for item in config.triggers_levels:
            if item[0] == "StagedTrainingTrigger":
                levels = item[1]
                break
        if not levels:
            logging.error("Error: No levels were successfully parsed!")
            return []

        allowed_keys = set(field.name for field in dataclasses.fields(StagedTrainingLevel))

        parsed_levels = []
        for level in levels:
            assert isinstance(level, dict)
            attrs = {}
            for key, val in level.items():
                if key not in allowed_keys:
                    logging.error(f"Unknown `StagedTrainingLevel` attribute: \"{key}\"")
                else:
                    attrs[key] = val

            # Check if all attributes are specified
            diff = allowed_keys - set(attrs.keys())
            if diff:
                for x in diff:
                    logging.error(f"Unspecified attribute: \"{x}\"")

            # Construct `StagedTrainingLevel` object
            parsed_levels.append(StagedTrainingLevel(**attrs))

        return parsed_levels



class TestEtaStatesTrigger:

    def __init__(self, config, agent, env: QEnv):
        self.config = config
        self.agent = agent
        self.env = env
        self._parse_etas(config)

    def __call__(self, iter: int, *args, **kwargs):
        logging.info(f"\n[TestEtaStatesTrigger]")

        for ss in self.subsystem_sizes:
            pool = {}
            logging.info(f"\tSubsystem size = {ss}")
            for eta in self.etas:
                logging.info(f"\t\tEta = {eta:.3f}")
                ents = self.get_entanglements(eta, ss)
                pool[eta] = ents

            fig, ax = plt.subplots(figsize=(6, 6))
            xs = sorted(pool.keys())

            Y = []
            for x in xs:
                ys = np.max(pool[x], axis=1)
                ax.scatter(np.full(len(ys), x), ys, color="tab:blue", s=10, alpha=0.3, ec=None)
                Y.append(ys)
            Y = np.atleast_2d(np.array(Y))
            envelope = np.max(Y, axis=1)

            ax.plot(xs, envelope, color="tab:red", ls='--', label="maximum")
            ax.plot(xs, np.percentile(Y, 34, axis=1), color="tab:orange", lw=0.8, label=r"$\sigma$")
            ax.plot(xs, np.percentile(Y, 68, axis=1), color="tab:orange", lw=0.8)
            ax.plot(xs, np.median(Y, axis=1), color="tab:red", lw=2.0, label=r"median")
            ax.set_yscale("log")
            ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
            ax.axhline(1e-3, ls='--', lw=1, color='k', label=r"$\epsilon$")
            ax.grid(which="both", alpha=0.3)
            ax.legend()
            ax.set_title(f"TestEtaStatesTrigger, subsystem size = {ss}\niteration {iter}")
            savedir = get_logdir(self.config)
            fig.savefig(os.path.join(savedir, f"TestEtaStatesTrigger-{iter}-{ss}.png"), dpi=240)

    def get_entanglements(self, eta: float, subsystem_size: int, num_steps: int = 30):
        sgen = stategen.StateGenerator(
            stategen.sample_haar_generalized,
            self.env.num_qubits,
            sample_params=dict(
                min_subsystem_size=subsystem_size,
                max_subsystem_size=subsystem_size,
                min_eta=eta,
                max_eta=eta
            )
        )
        env = QEnv(
            num_qubits=         self.env.num_qubits,
            num_envs=           self.env.num_envs,
            epsi=               self.env.epsi,
            max_episode_steps=  1000,
            reward_fn=          self.config.reward_fn,
            obs_fn=             self.config.obs_fn,
            state_generator=    sgen,
            fast_ents=          self.env.simulator.fast_ents,
            fast_obs=           self.env.fast_obs,
            swaps=              self.env.simulator.swaps,
            device=             self.env.device
        )
        env.reset()

        o = env.obs_fn(env.simulator.states)
        for _ in range(num_steps):
            p = self.agent.policy(o.to(device=self.config.model_device))
            # acts = p.probs().cpu().numpy()
            probs = p.probs.cpu().numpy()
            a = np.argmax(probs, axis=1)
            o, _, _, _, _ = env.step(a, reset=False)
        return env.simulator.entanglements.numpy()

    def _parse_etas(self, config):
        for item in config.triggers_levels:
            if item[0] == "TestEtaStatesTrigger":
                self.etas = item[1]["etas"]
                self.subsystem_sizes = item[1]["subsystem_sizes"]
                return
        logging.error("Error: Unable to parse `TestEtaStatesTrigger` config")
        self.etas = []
        self.subsystem_sizes = []

    def state_dict(self):
        return {"type": str(type(self))}

    def load_state_dict(self):
        pass


class UnfreezePolicy:

    def __init__(self, config, agent, env):
        self.config = config
        self.agent = agent
        self.env = env
        self.milestone = config.pi_freeze
        self.iteration = 0
        self.frozen = True
        logging.info("Initialized UnfreezePolicy trigger")

    def __call__(self, iter: int, *args, **kwargs):
        if iter >= self.milestone:
            self.agent.freeze_pf = False
            self.frozen = False
            logging.info(f"\n[UnfreezePolicy] Policy function unfrozen.")
        else:
            logging.info(f"\n[UnfreezePolicy] Policy function still frozen.")

    def state_dict(self):
        return dict(
            type =              str(type(self)),
            milestone =         self.milestone,
            frozen =            self.frozen
        )

    def load_state_dict(self, state_dict):
        self.frozen = state_dict["frozen"]
        self.milestone = state_dict["milestone"]
        self.current_level = state_dict["current_level"]
