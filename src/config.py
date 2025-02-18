import os

from yacs.config import CfgNode


DEFAULT = CfgNode()


# ===----------------- Environment related -----------------=== #
# Number of qubits
DEFAULT.num_qubits = 4

# Number of iterations in the environment loop
DEFAULT.num_iters = 1001

# Number of parallel sub-environments
DEFAULT.num_envs = 128

# Number of episode steps
DEFAULT.steps = 64

# Maximum steps before truncating an environment
DEFAULT.steps_limit = 40

# Threshold below which, the state is considered disentangled
DEFAULT.epsi = 1e-2

# Observation function used by the RL environment
DEFAULT.obs_fn = "rdm_2q_mean_real"

# Reward function used by the RL environment
DEFAULT.reward_fn = "relative_delta"

# State generator / sampler function used by the RL environment
DEFAULT.stategen_fn = "haar_product"

# State generator paramaters
DEFAULT.stategen_params = [("min_subsystem_size", 2), ("max_subsystem_size", 4)]


# ===----------------- Agent related -----------------=== #

# Policy network learning rate
DEFAULT.pi_lr = 2e-4

# Value network learning rate
DEFAULT.vf_lr = 3e-4

# Discount factor
DEFAULT.discount = 1.0

# Batch size for PPO
DEFAULT.batch_size = 512

# Clip value for gradient clipping by norm
DEFAULT.clip_grad = 1.0

# Entropy regularization parameter
DEFAULT.entropy_reg = 0.1

# Embedding dimension for self-attention keys, queries and values
DEFAULT.embed_dim = 256

# MLP dimension in Transformer encoder
DEFAULT.dim_mlp = 256

# Number of attention heads
DEFAULT.attn_heads = 4

# Number of transformer layers
DEFAULT.transformer_layers = 4


# ===----------------- Other -----------------=== #

DEFAULT.device = "cpu"

# Seed
DEFAULT.seed = 0

DEFAULT.log_every = 100

DEFAULT.checkpoint_every = 100

DEFAULT.test_every = 1000

DEFAULT.num_tests = 100

DEFAULT.trigger_every = 500

DEFAULT.triggers = ["StagedStateGeneratorTrigger"]

DEFAULT.reset_optimizers = False

DEFAULT.reset_tracker = False

DEFAULT.logdir_prefix = ""

DEFAULT.logdir_suffix = ""

DEFAULT.agent_checkpoint = ""

# ! This attribute is applicable only when the RL agent is evaluated / tested !
# If `True`, then action selection is performed using `argmax` over the RL agent's policy
# If `False`, then action is sampled using the policy's probabilities
DEFAULT.greedy_evaluation_policy = True



def get_default_config():
    return DEFAULT


def get_logdir(config):
    logdir_name = f"{config.num_qubits}q_{config.num_iters}iters"
    logdir_name = config.logdir_prefix + logdir_name + config.logdir_suffix
    logdir_path = os.path.join("logs", logdir_name)
    return logdir_path