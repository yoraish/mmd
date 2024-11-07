import os
import time

import torch.cuda
import wandb

from experiment_launcher import run_experiment, single_experiment_yaml


# This decorator creates results_dir as results_dir/seed, and saves the experiment arguments into a file.
@single_experiment_yaml
def experiment(
    #######################################
    env: str = 'env-name',  # You need to specify the argument type if you use the automatic parser.
    tensor_size: int = 10000,

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

    tensor1 = torch.ones(tensor_size, device='cuda' if torch.cuda.is_available() else 'cpu')
    tensor2 = tensor1.clone()

    print(f'Env: {env}')
    print(f'Tensor shape: {tensor1.shape}')

    filename = os.path.join(results_dir, 'log_' + str(seed) + '.txt')
    out_str = f'Running experiment with seed {seed}'
    with open(filename, 'w') as file:
        file.write('Some logs in a log file.\n')
        file.write(out_str)

    wandb.log({'seed': seed}, step=1)

    end_time = time.time() + 0.01
    while time.time() < end_time:
        tensor = tensor1 * tensor2
    print('Finished experiment.')


if __name__ == '__main__':
    # Leave unchanged
    run_experiment(experiment)
