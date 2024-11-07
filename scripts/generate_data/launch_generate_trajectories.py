import os
import socket

import numpy as np

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local

########################################################################################################################
# EXPERIMENT PARAMETERS SETUP
# SELECT ONE
# num_contexts: the number of start/goal pairs.
# num_trajectories_per_context: the number of trajectories per start/goal pair.

env_id: str = 'EnvHighways2D'
robot_id: str = 'RobotPlanarDisk'
num_contexts = 500
num_trajectories_per_context = 20
threshold_start_goal_pos: float = 0.9
is_start_goal_near_limits: bool = False
obstacle_cutoff_margin: float = 0.05

########################################################################################################################
# LAUNCHER

hostname = socket.gethostname()

LOCAL = is_local()
TEST = False
# USE_CUDA = True
USE_CUDA = False

N_SEEDS = num_contexts

N_EXPS_IN_PARALLEL = 6 if not USE_CUDA else 1

# N_CORES = N_EXPS_IN_PARALLEL
N_CORES = 8
MEMORY_SINGLE_JOB = 12000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'gpu' if USE_CUDA else 'amd3,amd2,amd'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1, gpu:rtx3090:1, gpu:a5000:1
CONDA_ENV = 'mmd'

exp_name = f'generate_trajectories'

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

launcher = Launcher(
    exp_name=exp_name,
    exp_file='generate_trajectories',
    n_seeds=N_SEEDS,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=0,
    hours=7,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    conda_env=CONDA_ENV,
    gres=GRES,
    use_timestamp=True
)


########################################################################################################################
# RUN

launcher.add_experiment(
    env_id__=env_id,
    robot_id__=robot_id,

    num_trajectories=num_trajectories_per_context,

    threshold_start_goal_pos=threshold_start_goal_pos,
    obstacle_cutoff_margin=obstacle_cutoff_margin,
    is_start_goal_near_limits=is_start_goal_near_limits,

    device='cuda' if USE_CUDA else 'cpu',

    debug=False
)

launcher.run(LOCAL, TEST)
