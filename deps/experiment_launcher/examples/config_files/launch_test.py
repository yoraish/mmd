from experiment_launcher import Launcher, is_local

LOCAL = is_local()
TEST = False
USE_CUDA = False

N_SEEDS = 3

if LOCAL:
    N_EXPS_IN_PARALLEL = 5
else:
    N_EXPS_IN_PARALLEL = 3

N_CORES = N_EXPS_IN_PARALLEL
MEMORY_SINGLE_JOB = 1000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'amd2,amd'  # 'amd', 'rtx'
GRES = 'gpu:1' if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = 'el'  # None

launcher = Launcher(
    exp_name='test_launcher',
    exp_file='test',
    # project_name='project01234',  # for hrz cluster
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
    use_timestamp=True,
    compact_dirs=False
)

config_files_l = [
    'configs/config00.yaml',
    'configs/config01.yaml',
]

# Optional arguments for Weights and Biases
wandb_options = dict(
    wandb_mode='disabled',  # "online", "offline" or "disabled"
    wandb_entity='joaocorreiacarvalho',
    wandb_project='test_experiment_launcher_config_files'
)

for i, config_file in enumerate(config_files_l):
    launcher.add_experiment(
        # A subdirectory will be created for parameters with a trailing double underscore.
        config__=f'config-{str(i).zfill(len(config_files_l))}',

        config_file_path=config_file,

        debug=False,

        **wandb_options,
        wandb_group=f'test_group-el-{config_file}'
    )

launcher.run(LOCAL, TEST)
