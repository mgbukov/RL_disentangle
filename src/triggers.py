import dataclasses
import logging
import numpy as np
from typing import *

from . import metrics
from . import stategen
from .evaluation import test_lengths


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

        logging.info(f"\n\n[StagedTrainingTrigger] Level {self.current_level}")

        if self.condition():
            self.action()
        else:
            logging.info("\n\tCondition not satisfied. "
                         f"Staying on level {self.current_level}")

    def condition(self):
        # Get current level parameters
        level = self.levels[self.current_level]

        # Initialize state sampler to test agent accuracy on the current level
        test_sampler = stategen.StateGenerator(
            sample_fn=stategen.sample_haar_product,
            num_qubits=self.config.num_qubits,
            sample_params=dict(
                min_subsystem_size=level.test_min_subsystem_size,
                max_subsystem_size=level.test_max_subsystem_size
            )
        )
        logging.info("\tInitialized Haar product state sampler with parameters:\n"
                     f"\t\tmin_subsystem_size = {level.test_min_subsystem_size}\n"
                     f"\t\tmax_subsystem_size = {level.test_max_subsystem_size}")


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
        if np.isnan(avg_length):
            avg_length = np.inf
        logging.info(f"\tRatio terminated: {ratio_terminated:.3f} "
                     f"({level.threshold_ratio_terminated:.2f})")
        logging.info(f"\tAverage length:   {avg_length:.1f} "
                     f"({level.threshold_average_length:.2f})")

        tracker = metrics.getTracker()
        tracker.add_scalar("[StagedTrainingTrigger] Max Subsystem Size", level.test_max_subsystem_size)
        tracker.add_scalar("[StagedTrainingTrigger] Ratio Terminated", ratio_terminated)
        tracker.add_scalar("[StagedTrainingTrigger] Average Length", avg_length)

        # Test condition
        if (avg_length <= level.threshold_average_length and
            ratio_terminated >= level.threshold_ratio_terminated):
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
            setattr(self.env.simulator.state_generator, name, val)
            newval = getattr(self.env.simulator.state_generator, name)
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

        allowed_keys = {
            "test_max_steps",
            "test_min_subsystem_size",
            "test_max_subsystem_size",
            "threshold_average_length",
            "threshold_ratio_terminated",
            "env_parameters",
            "sgen_parameters",
            "agent_parameters"
        }

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


class StagedStateGeneratorTrigger:
    """
    Changes the parameters of the state sampler
    """

    # Stages for the environment's stage generator
    sgen_stages = [
        {"min_subsystem_size": 1, "max_subsystem_size": 2},
        {"min_subsystem_size": 1, "max_subsystem_size": 3},
        {"min_subsystem_size": 1, "max_subsystem_size": 4},
        {"min_subsystem_size": 2, "max_subsystem_size": 5},
        {"min_subsystem_size": 2, "max_subsystem_size": 6},
    ]

    # Stages for the environment's `max_episode_length`
    eplen_stages = [
        10,
        10,
        10,
        35,
        80
    ]

    # Episode lenght condition will be multiplied by the number of qubits
    conditions = [
        {"subsystem_size": 2, "ratio_terminated": .98, "episode_length": 0.55},
        {"subsystem_size": 3, "ratio_terminated": .98, "episode_length": 0.7},
        {"subsystem_size": 4, "ratio_terminated": .98, "episode_length": 1.1},
        {"subsystem_size": 5, "ratio_terminated": .98, "episode_length": 4.3}, # 3.7 (6q)
        {"subsystem_size": 6, "ratio_terminated": .98, "episode_length": 60.0},
    ]

    def __init__(self, config, agent, env):
        self.config = config
        self.agent = agent
        self.environment = env
        self.current_stage = 0
        self.final_stage = len(type(self).conditions) - 1

    def condition(self):
        logging.info(f"\tCurrent stage: {self.current_stage}")
        if self.current_stage == self.final_stage:
            logging.info("\nAlready at the final stage.")
            return False

        conds = type(self).conditions[self.current_stage]

        # Initialize state sampler to test agent accuracy on the current stage
        test_sampler = stategen.StateGenerator(
            stategen.sample_haar_product,
            self.config.num_qubits,
            {"min_subsystem_size": conds["subsystem_size"],
             "max_subsystem_size": conds["subsystem_size"],
            }
        )
        logging.info("\tInitialized Haar product state sampler with parameters:\n"
                     f"\t\tmin_subsystem_size = {conds['subsystem_size']}\n"
                     f"\t\tmax_subsystem_size = {conds['subsystem_size']}")

        # What sould be the average steps that the agent takes on disentangling
        # a test state, before we transtition to the next stage of state sampling
        # during training ?
        avg_length_req = self.config.num_qubits * conds["episode_length"]
        # Limit on the maximum number of rollout steps during testing
        max_steps = int(2 * avg_length_req)
        logging.info(f"\tMaximum rollout steps: {max_steps}")

        # Test the agent
        lengths = test_lengths(
            self.agent,
            test_sampler,
            max_steps,
            obs_fn=self.config.obs_fn,
            num_tests=self.config.num_tests,
            epsi=self.config.epsi,
            greedy=self.config.greedy_evaluation_policy
        )

        # Calculate ration of disentangled states and average solution length
        ratio_terminated = np.sum(~np.isnan(lengths)) / lengths.size
        avg_length = np.nanmean(lengths)
        if np.isnan(avg_length):
            avg_length = np.inf
        ratio_terminated_req = conds["ratio_terminated"]
        logging.info(f"\tRatio terminated: {ratio_terminated:.3f} ({ratio_terminated_req:.2f})")
        logging.info(f"\tAverage length:   {avg_length:.1f} ({avg_length_req:.2f})")

        # Test condition
        if avg_length <= avg_length_req and ratio_terminated >= ratio_terminated_req:
            return True
        return False

    def action(self):
        if self.current_stage == self.final_stage:
            return
        # Transition
        self.current_stage += 1

        # Update state generator
        params = type(self).sgen_stages[self.current_stage]
        self.environment.simulator.state_generator.update(**params)

        # Update `max_episode_steps` and reset environment
        self.environment.max_episode_steps = type(self).eplen_stages[self.current_stage]
        self.environment.reset()

        # Log message
        params["max_episode_steps"] = type(self).eplen_stages[self.current_stage]
        msg = ("\n\tTransitioning to next stage with parameters:\n"
               + '\n'.join(f"\t\t{k}: {v}" for k, v in params.items()) )
        logging.info(msg)

    def __call__(self, *args, **kwargs):
        logging.info("\n\n[StagedStateSamplerTrigger]")
        if self.condition():
            self.action()
        else:
            if self.current_stage != self.final_stage:
                logging.info("\n\tCondition not satisfied. "
                             "Continuing with the current stage...")

