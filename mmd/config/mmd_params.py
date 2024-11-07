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
import torch


# A central location for aggregating the parameters used across files.
class MMDParams:
    # Robot parameters.
    robot_planar_disk_radius = 0.05
    # Single-agent planning parameters.
    use_guide_on_extra_objects_only = False
    n_samples = 64  # Batch size. Number of trajectories generated together.
    horizon = 64  # Number of steps in the trajectory.
    n_local_inference_noising_steps = 3  # Number of noising steps in local inference.
    n_local_inference_denoising_steps = 3  # Number of denoising steps in local inference.
    start_guide_steps_fraction = 0.5  # The fraction of the inference steps that are guided.
    n_guide_steps = 20  # The number of steps taken when applying conditioning at one diffusion step.
    n_diffusion_steps_without_noise = 1  # How many (at the end) diffusion steps get zero noise and guiding.
    weight_grad_cost_collision = 2e-2
    weight_grad_cost_smoothness = 8e-2
    weight_grad_cost_constraints = 2e-1
    weight_grad_cost_soft_constraints = 2e-2
    factor_num_interpolated_points_for_collision = 1.5
    trajectory_duration = 5.0
    device = 'cuda'
    debug = True
    seed = 18
    results_dir = 'logs'

    # Multi-agent planning parameters.
    vertex_constraint_radius = robot_planar_disk_radius * 2.4
    low_level_choose_path_from_batch_strategy = 'least_collisions'  # 'least_collisions' or 'least_cost'.

    # Evaluation.
    runtime_limit = 60  # 1 minute.
    data_adherence_linear_deviation_fraction = 0.1  # Points closer to start-goal line than fraction * length adhere.

    # Torch.
    tensor_args = {'device': device, 'dtype': torch.float32}

    # Trained models directory.
    trained_models_dir_global_fpath = '~/mmd/data_trained_models'  # Modify this.

