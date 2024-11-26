"""
Adapted from https://github.com/jacarvalho/mpd-public
"""
import os
from itertools import product

from experiment_launcher import Launcher
from experiment_launcher.utils import is_local

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

########################################################################################################################
# LAUNCHER

LOCAL = is_local()
TEST = False
USE_CUDA = True

N_SEEDS = 1

N_EXPS_IN_PARALLEL = 4

N_CORES = N_EXPS_IN_PARALLEL * 4
MEMORY_SINGLE_JOB = 12000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'gpu' if USE_CUDA else 'amd3,amd2,amd'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1, gpu:rtx3090:1, gpu:a5000:1
CONDA_ENV = 'mmd'


exp_name = f'train_diffusion'

launcher = Launcher(
    exp_name=exp_name,
    exp_file='train',
    # project_name='project01234',
    n_seeds=N_SEEDS,
    n_exps_in_parallel=N_EXPS_IN_PARALLEL,
    n_cores=N_CORES,
    memory_per_core=MEMORY_PER_CORE,
    days=2,
    hours=23,
    minutes=59,
    seconds=0,
    partition=PARTITION,
    conda_env=CONDA_ENV,
    gres=GRES,
    use_timestamp=True
)

########################################################################################################################
# EXPERIMENT PARAMETERS SETUP

dataset_subdir_l = [
    # 'EnvEmpty2D-RobotPlanarDisk',
    # 'EnvConveyor2D-RobotPlanarDisk',
    # 'EnvDropRegion2D-RobotPlanarDisk',
    # 'EnvHighways2D-RobotPlanarDisk',
    'EnvEmptyNoWait2D-RobotPlanarDisk',

    # 'EnvEmptyNoWait2D-RobotCompositeTwoPlanarDisk',
    # 'EnvEmptyNoWait2D-RobotCompositeThreePlanarDisk',
    # 'EnvEmptyNoWait2D-RobotCompositeSixPlanarDisk',
    # 'EnvEmptyNoWait2D-RobotCompositeNinePlanarDisk',
    # 'EnvHighways2D-RobotCompositeTwoPlanarDisk',
    # 'EnvHighways2D-RobotCompositeThreePlanarDisk',
    # 'EnvHighways2D-RobotCompositeSixPlanarDisk',
    # 'EnvHighways2D-RobotCompositeNinePlanarDisk',
]

include_velocity_l = [
    True
]

use_ema_l = [
    True
]

variance_schedule_l = [
    'exponential'
]

n_diffusion_steps_l = [
    25,
]

predict_epsilon_l = [
    True
]

dim = 32

unet_dim_mults_option_l = [
    0,
    # 1
]


batch_size = 128
lr = 3e-4


wandb_options = dict(
    wandb_mode='online',  # "online", "offline" or "disabled"
    wandb_entity='wandb_username',  # "username"
    wandb_project=exp_name
)

########################################################################################################################
# RUN

for dataset_subdir, include_velocity, use_ema, variance_schedule, n_diffusion_steps, predict_epsilon, unet_dim_mults_option in \
        product(dataset_subdir_l, include_velocity_l, use_ema_l, variance_schedule_l, n_diffusion_steps_l, predict_epsilon_l, unet_dim_mults_option_l):

    launcher.add_experiment(
        dataset_subdir__=dataset_subdir,
        include_velocity__=include_velocity,
        use_ema__=use_ema,
        variance_schedule__=variance_schedule,
        n_diffusion_steps__=n_diffusion_steps,
        predict_epsilon__=predict_epsilon,
        unet_dim_mults_option__=unet_dim_mults_option,

        lr=lr,

        batch_size=batch_size,
        num_train_steps=500000,

        steps_til_ckpt=20000,
        steps_til_summary=20000,

        **wandb_options,
        wandb_group=f'{dataset_subdir}-{include_velocity}-{use_ema}-{variance_schedule}-{n_diffusion_steps}-{predict_epsilon}-{unet_dim_mults_option}',

        debug=False,
    )

launcher.run(LOCAL, TEST)




