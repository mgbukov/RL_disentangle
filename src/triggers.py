import logging
import numpy as np

from . import stategen
from .evaluation import test_lengths


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

