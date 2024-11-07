"""
MIT License

Copyright (c) 2024 Yorai Shaoul

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# Standard imports.

# Project includes.
from mmd.common.experiments import MultiAgentPlanningExperimentConfig
from mmd.common.experiments.experiment_utils import *
from mmd.config.mmd_params import MMDParams as params
from launch_multi_agent_experiment import run_multi_agent_experiment


if __name__ == "__main__":
    # General:
    stagger_start_time_dt = 0
    runtime_limit = params.runtime_limit * 4
    num_trials_per_combination = 1
    render_animation = False
    experiment_instance_names = [
        "EnvEmpty2DRobotPlanarDiskCircle",
        "EnvEmpty2DRobotPlanarDiskBoundary"
    ]

    # Set up the MAPF experiments.
    # ==================================
    # Set up MMD MAPF experiments.
    # ==================================
    for instance_name in experiment_instance_names:
        # Set the torch random seed.
        torch.manual_seed(0)
        # Create an experiment config.
        experiment_config = MultiAgentPlanningExperimentConfig()
        # Set the experiment config.
        experiment_config.num_agents_l = [2*i for i in range(1, 16)]
        experiment_config.instance_name = instance_name

        experiment_config.stagger_start_time_dt = stagger_start_time_dt
        experiment_config.multi_agent_planner_class_l = ["XECBS", "ECBS", "PP", "CBS", "XCBS"]
        experiment_config.single_agent_planner_class = "MPDEnsemble"
        experiment_config.runtime_limit = runtime_limit
        experiment_config.num_trials_per_combination = num_trials_per_combination
        experiment_config.render_animation = render_animation
        # Run the experiment.
        run_multi_agent_experiment(experiment_config)
