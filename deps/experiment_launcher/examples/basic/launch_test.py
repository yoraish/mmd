from itertools import product

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
    gres=GRES,
    conda_env=CONDA_ENV,
    use_timestamp=True,
    compact_dirs=False
)

envs = {
    'env_00': {'env_param': 'aa'},
    'env_01': {'env_param': 'bb'}
}
a_l = [1, 2, 3]
boolean_param_l = [True, False]
some_default_param = 'b'

# These arguments are kwargs of the experiment function
unknown_args_list = [
    dict(integer_arg=10),
    # dict(floating_arg=11.0, string_arg='test')
]

for env in envs:
    d = envs[env]
    for a, boolean_param in product(a_l, boolean_param_l):
        for unknown_args in unknown_args_list:
            launcher.add_experiment(
                # A subdirectory will be created for parameters with a trailing double underscore.
                env__=env,
                a__=a,
                boolean_param__=boolean_param,

                env='some_env',  # This value will be overwritten by env__

                **d,
                some_default_param=some_default_param,

                **unknown_args,
                debug=False,
            )

launcher.run(LOCAL, TEST)
