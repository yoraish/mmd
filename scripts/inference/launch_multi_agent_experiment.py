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
from mmd.common.experiments.experiment_utils import *
from inference_multi_agent import run_multi_agent_trial


def run_multi_agent_experiment(experiment_config: MultiAgentPlanningExperimentConfig):
    # Run the multi-agent planning experiment.
    startt = time.time()
    # Create the experiment config.
    experiment_config.time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Get single trial configs from the experiment config.
    single_trial_configs = experiment_config.get_single_trial_configs_from_experiment_config()
    # So let's run sequentially.
    for single_trial_config in single_trial_configs:
        print(single_trial_config)
        try:
            run_multi_agent_trial(single_trial_config)

            # Aggregate and save data on every step. This is not needed (can be done once at the end).
            combine_and_save_results_for_experiment(experiment_config)
        except Exception as e:
            print("Error in run_multi_agent_experiment: ", e)
            # Save to a file.
            with open(f"error_{experiment_config.time_str}.txt", "a") as f:
                f.write(str(e))
                f.write("This is for single_trial_config: ")
                f.write(str(single_trial_config))
                f.write("\n")
            continue

    # Print the runtime.
    print("Runtime: ", time.time() - startt)
    print("Run: OK.")


if __name__ == "__main__":

    # Instance names. These dictate the maps and start/goals.
    # Create an experiment config.
    experiment_config = MultiAgentPlanningExperimentConfig()
    # Set the experiment config.
    experiment_config.num_agents_l = [2, 3, 6]

    # Single tile.
    # experiment_config.instance_name = "EnvEmpty2DRobotPlanarDiskCircle"
    # experiment_config.instance_name = "EnvEmpty2DRobotPlanarDiskBoundary"
    # experiment_config.instance_name = "EnvConveyor2DRobotPlanarDiskBoundary"
    # experiment_config.instance_name = "EnvHighways2DRobotPlanarDiskSmallCircle"
    # experiment_config.instance_name = "EnvDropRegion2DRobotPlanarDiskBoundary"
    # experiment_config.instance_name = "EnvConveyor2DRobotPlanarDiskRandom"
    # experiment_config.instance_name = "EnvEmpty2DRobotPlanarDiskRandom"
    experiment_config.instance_name = "EnvHighways2DRobotPlanarDiskRandom"

    # Multiple tiles.
    # experiment_config.instance_name = "EnvTestTwoByTwoRobotPlanarDiskRandom"
    # experiment_config.instance_name = "EnvTestThreeByThreeRobotPlanarDiskRandom"
    # experiment_config.instance_name = "EnvTestFourByFourRobotPlanarDiskRandom"

    experiment_config.stagger_start_time_dt = 0
    experiment_config.multi_agent_planner_class_l = ["XECBS"]  # , "ECBS", "PP", "XCBS", "CBS"]
    experiment_config.single_agent_planner_class = "MPDEnsemble"
    experiment_config.runtime_limit = 60 * 3
    experiment_config.num_trials_per_combination = 1
    experiment_config.render_animation = True
    # Run the experiment.
    run_multi_agent_experiment(experiment_config)

    # Example for only combining results without running the experiment.
    # experiment_config.time_str = "2024-09-08-13-55-05"
    # # Get all the results.
    # aggregated_results = combine_and_save_results_for_experiment(experiment_config)
