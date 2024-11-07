import os

import wandb
import yaml

from experiment_launcher import run_experiment, single_experiment_yaml


# This decorator creates results_dir as results_dir/seed, and saves the experiment arguments into a file.
@single_experiment_yaml
def experiment(
    #######################################
    config_file_path: str = './configs/config00.yaml',

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

    with open(config_file_path, 'r') as f:
        configs = yaml.load(f, yaml.Loader)

    print('Config file content:')
    print(configs)

    filename = os.path.join(results_dir, 'log_' + str(seed) + '.txt')
    out_str = f'Running experiment with seed {seed}'
    with open(filename, 'w') as file:
        file.write('Some logs in a log file.\n')
        file.write(out_str)

    wandb.log({'seed': seed}, step=1)


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)
