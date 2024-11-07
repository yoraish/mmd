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
import os
import pickle
from datetime import datetime
import time
from enum import Enum
from math import ceil
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
from typing import Tuple, List
import concurrent.futures
# Project imports.
from mmd.common.experiments import *


def read_aggregated_trial_results_for_experiment(experiment_config: MultiAgentPlanningExperimentConfig) -> dict:
    # Read all results for an experiment. Return a dictionary of the form:
    # {instance_name:
    #     {num_agents:
    #           {multi_agent_planner_class:
    #              NO DISTINCTION BETWEEN SINGLE AGENT PLANNERS FOR NOW.
    #                   [results]
    #  }}}}
    results_dir = get_result_dir_from_time_str(experiment_config.time_str)
    # Get all the results.
    aggregated_results = {}
    instance_name = experiment_config.instance_name
    for num_agents in experiment_config.num_agents_l:
        aggregated_results[num_agents] = {}
        for multi_agent_planner_class in experiment_config.multi_agent_planner_class_l:
            aggregated_results[num_agents][multi_agent_planner_class] = []
            for trial_number in range(experiment_config.num_trials_per_combination):
                trial_config = MultiAgentPlanningSingleTrialConfig()
                trial_config.instance_name = instance_name
                trial_config.num_agents = num_agents
                trial_config.multi_agent_planner_class = multi_agent_planner_class
                trial_config.single_agent_planner_class = experiment_config.single_agent_planner_class
                results_dir = get_result_dir_from_trial_config(trial_config,
                                                               time_str=experiment_config.time_str,
                                                               trial_number=trial_number)
                # Read the results.
                trial_result_fpath = os.path.join(results_dir, 'results.pkl')
                # Check that this file exists.
                if not os.path.exists(trial_result_fpath):
                    # print(f"Warning: {trial_result_fpath} does not exist.")
                    continue

                with open(trial_result_fpath, 'rb') as f:
                    trial_result = pickle.load(f)  # Object type is MultiAgentPlanningSingleTrialResult.
                aggregated_results[num_agents][multi_agent_planner_class].append(trial_result)

    return aggregated_results


def combine_and_save_results_for_experiment(experiment_config: MultiAgentPlanningExperimentConfig):
    # Analyze all results for an experiment. Returns a dictionary of the form:
    # {instance_name:
    #     {num_agents:
    #           {multi_agent_planner_class:
    #              avg_ct_expansions,
    #              success_rate,
    #              fail_rate_runtime_limit,
    #              fail_rate_no_solution,
    #              avg_num_collisions_in_solution,
    #              avg_data_adherence,
    #              avg_planning_time,
    #              avg_path_length_per_agent,
    #              avg_mean_path_acceleration_agent_avg
    #  }}}}
    aggregated_results = read_aggregated_trial_results_for_experiment(experiment_config)
    # Analyze the results.
    analyzed_results = {}
    for num_agents in experiment_config.num_agents_l:
        analyzed_results[num_agents] = {}
        for multi_agent_planner_class in experiment_config.multi_agent_planner_class_l:
            analyzed_results[num_agents][multi_agent_planner_class] = {}
            # Keep track of the number of successful trials to normalize some metrics based on that.
            num_successful_trials = 0
            for trial_result in aggregated_results[num_agents][multi_agent_planner_class]:
                if trial_result.success_status == TrialSuccessStatus.SUCCESS:
                    num_successful_trials += 1
                # Initialize the dictionary.
                if 'avg_ct_expansions' not in analyzed_results[num_agents][multi_agent_planner_class]:
                    analyzed_results[num_agents][multi_agent_planner_class]['avg_data_adherence'] = 0.0
                    analyzed_results[num_agents][multi_agent_planner_class]['avg_planning_time'] = 0.0
                    analyzed_results[num_agents][multi_agent_planner_class]['avg_path_length_per_agent'] = 0.0
                    analyzed_results[num_agents][multi_agent_planner_class]['avg_mean_path_acceleration_per_agent'] = 0.0
                    analyzed_results[num_agents][multi_agent_planner_class]['success_rate'] = 0.0
                    analyzed_results[num_agents][multi_agent_planner_class]['fail_rate_runtime_limit'] = 0.0
                    analyzed_results[num_agents][multi_agent_planner_class]['fail_rate_no_solution'] = 0.0
                    analyzed_results[num_agents][multi_agent_planner_class]['fail_rate_collision_agents'] = 0.0
                    analyzed_results[num_agents][multi_agent_planner_class]['avg_num_collisions_in_solution'] = 0.0
                    analyzed_results[num_agents][multi_agent_planner_class]['avg_ct_expansions'] = 0.0

                # Add the metrics that need to be normalized by the total number of trials.
                analyzed_results[num_agents][multi_agent_planner_class][
                    'success_rate'] += 1 if trial_result.success_status == TrialSuccessStatus.SUCCESS else 0
                analyzed_results[num_agents][multi_agent_planner_class][
                    'fail_rate_runtime_limit'] += 1 if trial_result.success_status == TrialSuccessStatus.FAIL_RUNTIME_LIMIT else 0
                analyzed_results[num_agents][multi_agent_planner_class][
                    'fail_rate_no_solution'] += 1 if trial_result.success_status == TrialSuccessStatus.FAIL_NO_SOLUTION else 0
                analyzed_results[num_agents][multi_agent_planner_class][
                    'fail_rate_collision_agents'] += 1 if trial_result.success_status == TrialSuccessStatus.FAIL_COLLISION_AGENTS else 0

                # Add the metrics that need to be normalized by the total number of successful trials.
                if trial_result.success_status == TrialSuccessStatus.SUCCESS:
                    analyzed_results[num_agents][multi_agent_planner_class][
                        'avg_ct_expansions'] += trial_result.num_ct_expansions
                    analyzed_results[num_agents][multi_agent_planner_class][
                        'avg_data_adherence'] += trial_result.data_adherence
                    analyzed_results[num_agents][multi_agent_planner_class][
                        'avg_planning_time'] += trial_result.planning_time
                    analyzed_results[num_agents][multi_agent_planner_class][
                        'avg_path_length_per_agent'] += trial_result.path_length_per_agent
                    analyzed_results[num_agents][multi_agent_planner_class][
                        'avg_mean_path_acceleration_per_agent'] += trial_result.mean_path_acceleration_per_agent
                    analyzed_results[num_agents][multi_agent_planner_class][
                        'avg_num_collisions_in_solution'] += trial_result.num_collisions_in_solution

            # Normalize the dictionary.
            num_trials = len(aggregated_results[num_agents][multi_agent_planner_class])
            if num_trials > 0:
                # Some metrics normalized by num_trials.
                analyzed_results[num_agents][multi_agent_planner_class]['success_rate'] /= num_trials
                analyzed_results[num_agents][multi_agent_planner_class]['fail_rate_runtime_limit'] /= num_trials
                analyzed_results[num_agents][multi_agent_planner_class]['fail_rate_no_solution'] /= num_trials
                analyzed_results[num_agents][multi_agent_planner_class]['fail_rate_collision_agents'] /= num_trials
                # Normalize by the number of successful trials.
                if num_successful_trials > 0:
                    analyzed_results[num_agents][multi_agent_planner_class]['avg_ct_expansions'] /= num_successful_trials
                    analyzed_results[num_agents][multi_agent_planner_class]['avg_data_adherence'] /= num_successful_trials
                    analyzed_results[num_agents][multi_agent_planner_class]['avg_planning_time'] /= num_successful_trials
                    analyzed_results[num_agents][multi_agent_planner_class]['avg_path_length_per_agent'] /= num_successful_trials
                    analyzed_results[num_agents][multi_agent_planner_class]['avg_mean_path_acceleration_per_agent'] /= num_successful_trials
                    analyzed_results[num_agents][multi_agent_planner_class]['avg_num_collisions_in_solution'] /= num_trials

    # Save to a CSV in the results directory. Create one CSV per number of agents.
    results_dir = get_results_dir_from_experiment_config(experiment_config)
    df_num_agents_l = []
    for num_agents in experiment_config.num_agents_l:
        fpath_results_num_agents = os.path.join(results_dir, f'aggregated_results_{num_agents}_agents.csv')
        # If this already exists, delete it.
        if os.path.exists(fpath_results_num_agents):
            os.remove(fpath_results_num_agents)
        df = pd.DataFrame(analyzed_results[num_agents])
        # Add a num_agents row.
        df.loc['num_agents'] = num_agents
        # # Transpose the dataframe.
        # df = df.T
        df.to_csv(fpath_results_num_agents)
        df_num_agents_l.append(df.T)

    # Now combine all csv files into one. For each, transpose and add a num_agents column.
    fpath_results_all_agents = os.path.join(results_dir, 'aggregated_results_all_agents.csv')
    # If this already exists, delete it.
    if os.path.exists(fpath_results_all_agents):
        os.remove(fpath_results_all_agents)
    # Create a new dataframe.
    df_all_agents = pd.concat(df_num_agents_l)
    # This df treats the method names as index. We want them as a column with "method" as the column name.
    df_all_agents.reset_index(inplace=True)
    df_all_agents.rename(columns={'index': 'method'}, inplace=True)

    df_all_agents.to_csv(fpath_results_all_agents, index=False)
    print(f"Results saved to file://{fpath_results_all_agents}")

    return analyzed_results
