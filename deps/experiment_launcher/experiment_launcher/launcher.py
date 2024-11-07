import argparse
import datetime
import inspect
import os
import traceback
from copy import copy, deepcopy
from distutils.util import strtobool
from importlib import import_module

import numpy as np
from joblib import Parallel, delayed

from experiment_launcher.exceptions import ResultsDirException


class Launcher(object):
    """
    Creates and starts jobs with Joblib or SLURM.

    """

    def __init__(self, exp_name, exp_file, n_seeds, start_seed=0, n_cores=1, memory_per_core=2000,
                 days=0, hours=24, minutes=0, seconds=0,
                 project_name=None, base_dir=None,
                 n_exps_in_parallel=1,
                 conda_env=None, gres=None, constraint=None, partition=None,
                 begin=None, use_timestamp=True, compact_dirs=False):
        """
        Constructor.

        Args:
            exp_name (str): name of the experiment
            exp_file (str): name of the python module running a single experiment (relative path)
            n_seeds (int): number of seeds for each experiment configuration
            start_seed (int): first seed
            n_cores (int): number of cpu cores
            memory_per_core (int): maximum memory per core (slurm will kill the job if this is reached)
            days (int): number of days the experiment can last (in slurm)
            hours (int): number of hours the experiment can last (in slurm)
            minutes (int): number of minutes the experiment can last (in slurm)
            seconds (int): number of seconds the experiment can last (in slurm)
            project_name (str): name of the project for slurm. This is important if you have
                different projects (e.g. in the hhlr cluster)
            base_dir (str): path to directory to save the results (in hhlr results are saved to /work/scratch/$USER)
            n_exps_in_parallel (int): number of experiment configurations to run in parallel.
                If running in the cluster, and the gpu is selected, then it is the number of jobs in each slurm file
                (e.g. for multiple experiments in the same gpu)
            conda_env (str): name of the conda environment to run the experiments in
            gres (str): request cluster resources. E.g. to add a GPU in the IAS cluster specify gres='gpu:rtx2080:1'
            constraint (str): constraint for the slurm job. E.g. to add a GPU in the IAS cluster: constraint='rtx2080'
            partition (str, None): the partition to use in case of slurm execution. If None, no partition is specified.
            begin (str): start the slurm experiment at a given time (see --begin in slurm docs)
            use_timestamp (bool): add a timestamp to the experiment name
            compact_dirs (bool): If true, only the parameter value is used for the directory name.

        """
        self._exp_name = exp_name
        self._exp_file = exp_file
        self._start_seed = start_seed
        self._n_seeds = n_seeds
        self._n_cores = n_cores
        self._memory_per_core = memory_per_core
        self._duration = Launcher._to_duration(days, hours, minutes, seconds)
        self._project_name = project_name
        self._n_exps_in_parallel = n_exps_in_parallel
        self._conda_env = conda_env
        self._gres = gres
        self._constraint = constraint
        self._partition = partition
        self._begin = begin

        self._experiment_list = list()

        if use_timestamp:
            self._exp_name += datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')

        base_dir = './logs' if base_dir is None else base_dir
        self._exp_dir_local = os.path.join(base_dir, self._exp_name)

        # tracks the results directories
        self._results_dir_l = []

        self._exp_dir_slurm = self._exp_dir_local
        if os.getenv("USER"):
            scratch_dir = os.path.join('/work', 'scratch', os.getenv('USER'))
            if os.path.isdir(scratch_dir):
                self._exp_dir_slurm = os.path.join(scratch_dir, self._exp_name)
        os.makedirs(self._exp_dir_slurm)

        # directories for slurm sbatch files and logs
        self._exp_dir_slurm_files = os.path.join(self._exp_dir_slurm, "slurm_files")
        self._exp_dir_slurm_logs = os.path.join(self._exp_dir_slurm, "slurm_logs")

        self._compact_dirs = compact_dirs

    def add_experiment(self, **kwargs):
        self._experiment_list.append(deepcopy(kwargs))

    def run(self, local, test=False, sequential=False):
        self._check_experiments_results_directories()
        if local:
            if sequential:
                self._run_sequential(test)
            else:
                self._run_local(test)
        else:
            self._run_slurm(test)

        self._experiment_list = list()

    def generate_slurm(self, command_line_list=None):
        project_name_option = ''
        partition_option = ''
        begin_option = ''
        gres_option = ''
        constraint_option = ''

        if self._project_name:
            project_name_option = '#SBATCH -A ' + self._project_name + '\n'
        if self._partition:
            partition_option += f'#SBATCH -p {self._partition}\n'
        if self._begin:
            begin_option += f'#SBATCH --begin={self._begin}\n'
        if self._gres:
            print(self._gres)
            gres_option += '#SBATCH --gres=' + str(self._gres) + '\n'
        if self._constraint:
            print(self._constraint)
            constraint_option += '#SBATCH --constraint=' + str(self._constraint) + '\n'

        conda_code = ''
        if self._conda_env:
            if os.path.exists(f'{os.getenv("HOME")}/miniconda3'):
                conda_code += f'eval \"$({os.getenv("HOME")}/miniconda3/bin/conda shell.bash hook)\"\n'
            elif os.path.exists(f'{os.getenv("HOME")}/anaconda3'):
                conda_code += f'eval \"$({os.getenv("HOME")}/anaconda/bin/conda shell.bash hook)\"\n'
            else:
                raise Exception('You do not have a /home/USER/miniconda3 or /home/USER/anaconda3 directories')
            conda_code += f'conda activate {self._conda_env}\n\n'
            python_code = f'python {self._exp_file_path} \\'
        else:
            python_code = f'python3  {self._exp_file_path} \\'

        experiment_args = '\t\t'
        experiment_args += r'${@: 2}'
        experiment_args += ' \\'

        seed_code = f'\t\t--seed $(({self._start_seed} + $SLURM_ARRAY_TASK_ID)) \\'
        result_dir_code = '\t\t--results_dir $1'

        code = f"""\
#!/usr/bin/env bash

###############################################################################
# SLURM Configurations

# Optional parameters
{project_name_option}{partition_option}{begin_option}{gres_option}{constraint_option}
# Mandatory parameters
#SBATCH -J {self._exp_name}
#SBATCH -a 0-{self._n_seeds - 1}
#SBATCH -t {self._duration}
#SBATCH --ntasks 1
#SBATCH --cpus-per-task {self._n_cores}
#SBATCH --mem-per-cpu={self._memory_per_core}
#SBATCH -o {self._exp_dir_slurm_logs}/%A_%a.out
#SBATCH -e {self._exp_dir_slurm_logs}/%A_%a.err

###############################################################################
# Your PROGRAM call starts here
echo "Starting Job $SLURM_JOB_ID, Index $SLURM_ARRAY_TASK_ID"

# Program specific arguments
{conda_code}

"""
        code += f"""\
# Program specific arguments

echo "Running scripts in parallel..."
echo "########################################################################"
            
"""
        for command_line in command_line_list:
            code += f"""\
                
{python_code}
{seed_code}
\t\t{command_line}  &

"""

        code += f"""\
            
wait # This will wait until both scripts finish
echo "########################################################################"
echo "...done."
"""
        return code

    def save_slurm(self, command_line_list=None, idx: str = None):
        code = self.generate_slurm(command_line_list)

        label = f"_{idx}" if idx is not None else ""
        script_name = f'slurm_{self._exp_name}{label}.sh'
        full_path = os.path.join(self._exp_dir_slurm_files, script_name)

        with open(full_path, "w") as file:
            file.write(code)

        return full_path

    def _run_slurm(self, test):
        # Create slurm directories for sbatch and log files
        os.makedirs(self._exp_dir_slurm_files)
        os.makedirs(self._exp_dir_slurm_logs)

        # Generate and save slurm files
        slurm_files_path_l = []
        experiment_list_chunked = []
        for i in range(0, len(self._experiment_list), self._n_exps_in_parallel):
            experiment_list_chunked.append(self._experiment_list[i:i + self._n_exps_in_parallel])

        for i, exps in enumerate(experiment_list_chunked):
            command_line_l = []
            for exp in exps:
                exp_new_without_underscore = self.remove_last_underscores_dict(exp)
                command_line_arguments = self._convert_to_command_line(exp_new_without_underscore)
                results_dir = self._generate_results_dir(self._exp_dir_slurm, exp)
                command_line_l.append(f'--results_dir {results_dir} {command_line_arguments}')
            slurm_files_path_l.append(self.save_slurm(command_line_l,
                                                      str(i).zfill(len(str(len(experiment_list_chunked))))))

        # Launch slurm files in parallel
        for slurm_file_path in slurm_files_path_l:
            command = f"sbatch {slurm_file_path}"
            if test:
                print(command)
            else:
                os.system(command)

    def _run_local(self, test):
        if not test:
            os.makedirs(self._exp_dir_local, exist_ok=True)

        module = import_module(self._exp_file)
        experiment = module.experiment

        if test:
            self._test_experiment_local()
        else:
            def experiment_wrapper(params):
                try:
                    experiment(**params)
                except Exception:
                    print("Experiment failed with parameters:")
                    print(params)
                    traceback.print_exc()

            params_dict = get_experiment_default_params(experiment)

            Parallel(n_jobs=self._n_exps_in_parallel)(delayed(experiment_wrapper)(deepcopy(params))
                                                      for params in self._generate_exp_params(params_dict))

    def _run_sequential(self, test):
        if not test:
            os.makedirs(self._exp_dir_local, exist_ok=True)

        module = import_module(self._exp_file)
        experiment = module.experiment

        if test:
            self._test_experiment_local()
        else:
            default_params_dict = get_experiment_default_params(experiment)

            for params in self._generate_exp_params(default_params_dict):
                try:
                    experiment(**params)
                except Exception:
                    print("Experiment failed with parameters:")
                    print(params)
                    traceback.print_exc()

    def _check_experiments_results_directories(self):
        """
        Check if the results directory produced for each experiment clash.
        """
        for exp in self._experiment_list:
            results_dir = self._generate_results_dir(self._exp_dir_local, exp)
            # Check if the results directory already exists.
            if results_dir in self._results_dir_l:
                # Terminate to prevent overwriting the results directory.
                raise ResultsDirException(exp, results_dir)
            self._results_dir_l.append(results_dir)

    def _test_experiment_local(self):
        for exp, results_dir in zip(self._experiment_list, self._results_dir_l):
            for i in range(self._start_seed, self._n_seeds):
                params = str(exp).replace('{', '(').replace('}', '').replace(': ', '=').replace('\'', '')
                if params:
                    params += ', '
                print('experiment' + params + 'seed=' + str(i) + ', results_dir=' + results_dir + ')')

    def _generate_results_dir(self, results_dir, exp, n=6):
        for key, value in exp.items():
            if key.endswith('__'):
                if self._compact_dirs:
                    subfolder = str(value)
                else:
                    subfolder = key + '_' + str(value).replace(' ', '')
                subfolder = subfolder.replace('/', '-')  # avoids creating subfolders if there is a slash in the name
                results_dir = os.path.join(results_dir, subfolder)
        return results_dir

    def _generate_exp_params(self, default_params_dict):
        seeds = np.arange(self._start_seed, self._start_seed + self._n_seeds)
        for exp in self._experiment_list:
            params_dict = deepcopy(default_params_dict)
            exp_new_without_underscore = self.remove_last_underscores_dict(exp)
            params_dict.update(exp_new_without_underscore)
            params_dict['results_dir'] = self._generate_results_dir(self._exp_dir_local, exp)
            for seed in seeds:
                params_dict['seed'] = int(seed)
                yield params_dict

    @staticmethod
    def remove_last_underscores_dict(exp_dict):
        exp_dict_new = copy(exp_dict)
        for key, value in exp_dict.items():
            if key.endswith('__'):
                exp_dict_new[key[:-2]] = value
                del exp_dict_new[key]
        return exp_dict_new

    @staticmethod
    def _convert_to_command_line(exp):
        command_line = ''
        for key, value in exp.items():
            new_command = '--' + key + ' '

            if isinstance(value, list):
                new_command += ' '.join(map(str, value)) + ' '
            else:
                new_command += str(value) + ' '

            command_line += new_command

        # remove last space
        command_line = command_line[:-1]

        return command_line

    @staticmethod
    def _to_duration(days, hours, minutes, seconds):
        h = "0" + str(hours) if hours < 10 else str(hours)
        m = "0" + str(minutes) if minutes < 10 else str(minutes)
        s = "0" + str(seconds) if seconds < 10 else str(seconds)

        return str(days) + '-' + h + ":" + m + ":" + s

    @property
    def exp_name(self):
        return self._exp_name

    def log_dir(self, local=True):
        if local:
            return self._exp_dir_local
        else:
            return self._exp_dir_slurm

    @property
    def _exp_file_path(self):
        module = import_module(self._exp_file)
        return module.__file__


def get_experiment_default_params(func):
    signature = inspect.signature(func)
    defaults = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            defaults[k] = v.default
    return defaults


def translate_experiment_params_to_argparse(parser, func):
    annotation_to_argparse = {
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': None,
        'tuple': None,
    }
    arg_experiments = parser.add_argument_group('Experiment')
    signature = inspect.signature(func)
    for k, v in signature.parameters.items():
        if k not in ['seed', 'results_dir']:
            if v.default is not inspect.Parameter.empty:
                if v.annotation.__name__ in annotation_to_argparse:
                    if v.annotation.__name__ == 'bool':
                        arg_experiments.add_argument(f"--{str(k)}", type=lambda x: bool(strtobool(x)),
                                                     nargs='?', const=v.default, default=v.default)
                    elif v.annotation.__name__ == 'list':
                        arg_experiments.add_argument(f"--{str(k)}", nargs='+')
                    else:
                        arg_experiments.add_argument(f"--{str(k)}", type=annotation_to_argparse[v.annotation.__name__])
                else:
                    raise NotImplementedError(f'{v.annotation.__name__} not found in annotation_to_argparse.')
    return parser


def add_launcher_base_args(parser):
    arg_default = parser.add_argument_group('Default')
    arg_default.add_argument('--seed', type=int)
    arg_default.add_argument('--results_dir', type=str)
    return parser


def has_kwargs(func):
    signature = inspect.signature(func)
    for k, v in signature.parameters.items():
        if v.kind == v.VAR_KEYWORD:
            return True

    return False


def string_to_primitive(string):
    try:
        return int(string)
    except ValueError:
        try:
            return float(string)
        except ValueError:
            try:
                # boolean
                return bool(strtobool(string))
            except ValueError:
                return string


def parse_unknown_args(unknown):
    kwargs = dict()

    key_idxs = [i for i, arg in enumerate(unknown) if arg.startswith('--')]

    if len(key_idxs) > 0:
        key_n_args = [key_idxs[i+1] - 1 - key_idxs[i] for i in range(len(key_idxs)-1)]
        key_n_args.append(len(unknown) - 1 - key_idxs[-1])

        for i, idx in enumerate(key_idxs):
            key = unknown[idx][2:]
            n_args = key_n_args[i]
            if n_args > 1:
                values = list()

                for v in unknown[idx+1:idx + 1 + n_args]:
                    values.append(string_to_primitive(v))

                kwargs[key] = values

            elif n_args == 1:
                kwargs[key] = string_to_primitive(unknown[idx+1])

    return kwargs


def parse_args(func):
    parser = argparse.ArgumentParser()

    parser = translate_experiment_params_to_argparse(parser, func)

    parser = add_launcher_base_args(parser)
    parser.set_defaults(**get_experiment_default_params(func))

    if has_kwargs(func):
        args, unknown = parser.parse_known_args()
        kwargs = parse_unknown_args(unknown)

        args = vars(args)
        args.update(kwargs)

        return args
    else:
        args = parser.parse_args()
        return vars(args)


def run_experiment(func, args=None):
    if not args:
        args = parse_args(func)

    func(**args)
