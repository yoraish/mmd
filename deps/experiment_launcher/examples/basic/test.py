import os

import wandb

from experiment_launcher import run_experiment, single_experiment


# This decorator creates results_dir as results_dir/seed, and saves the experiment arguments into a file.
@single_experiment
def experiment(
    #######################################
    env: str = 'env',  # You need to specify the argument type if you use the automatic parser.
    env_param: str = 'aa',
    a: int = 1,
    boolean_param: bool = True,
    some_default_param: str = 'b',

    debug: bool = True,

    #######################################
    # MANDATORY
    seed: int = 0,
    results_dir: str = 'logs',

    #######################################
    # OPTIONAL
    # accept unknown arguments
    **kwargs
):
    # EXPERIMENT
    print(f'DEBUG MODE: {debug}')

    print(f'kwargs: {kwargs}')

    filename = os.path.join(results_dir, 'log_' + str(seed) + '.txt')
    out_str = f'Running experiment with seed {seed}, env {env}, ' \
              f'env_param {env_param}, a {a}, ' \
              f'boolean_param {boolean_param}, some_default_param {some_default_param}'
    print(out_str)
    with open(filename, 'w') as file:
        file.write('Some logs in a log file.\n')
        file.write(out_str)


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)
