from experiment_launcher import Launcher, is_local

LOCAL = is_local()
TEST = False
USE_CUDA = True

N_SEEDS = 10

if LOCAL:
    N_EXPS_IN_PARALLEL = 2
else:
    N_EXPS_IN_PARALLEL = 3

N_CORES = N_EXPS_IN_PARALLEL
MEMORY_SINGLE_JOB = 1000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = 'rtx2,rtx' if USE_CUDA else 'amd2,amd'
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

tensor_sizes_l = [1000000 + i for i in range(5)]

# Optional arguments for Weights and Biases
wandb_options = dict(
    wandb_mode='disabled',  # "online", "offline" or "disabled"
    wandb_entity='joaocorreiacarvalho',
    wandb_project='test_experiment_launcher_config_files'
)

for tensor_size in tensor_sizes_l:
    launcher.add_experiment(
        # A subdirectory will be created for parameters with a trailing double underscore.
        tensor_size__=tensor_size,

        debug=False,

        **wandb_options,
        wandb_group=f'test_group-el-{tensor_size}',
    )

launcher.run(LOCAL, TEST)
