The original repository is this one: https://github.com/anindex/torch_robotics.
Their original README follows.

# TorchRobotics

This library implements differentiable robot tree from URDF or MCJF robot format, and the differentiable planning objects such as obstacle avoidance, self-collision avoidance and via point.

**NOTE**: `torch_robotics` is under heavy development and highly experimental.

## Installation

Simply activate your conda/Python environment and run

```azure
pip install -e .
```

## Examples

To see FK, IK of all available robot kinematics

```azure
python examples/forward_kinematics.py
```

and

```azure
python examples/inverse_kinematics.py
```

## Acknowledgements

A part of this implementation is inspired from the library [differentiable robot model](https://github.com/facebookresearch/differentiable-robot-model).

## Contact

If you have any questions or find any bugs, please let us know:

- [An Le](https://www.ias.informatik.tu-darmstadt.de/Team/AnThaiLe), [an@robot-learning.de](an@robot-learning.de)
- [Joao Carvalho](https://www.ias.informatik.tu-darmstadt.de/Team/JoaoCarvalho), [joao@robot-learning.de](joao@robot-learning.de)

## Citation

If you found this repository useful, please consider citing these references:

```azure
@inproceedings{le2023accelerating,
  title={Accelerating Motion Planning via Optimal Transport},
  author={Le, An T. and Chalvatzaki, Georgia and Biess, Armin and Peters, Jan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}

@article{carvalho2023motion,
  title={Motion planning diffusion: Learning and planning of robot motions with diffusion models},
  author={Carvalho, Joao and Le, An T and Baierl, Mark and Koert, Dorothea and Peters, Jan},
  journal={arXiv preprint arXiv:2308.01557},
  year={2023}
}
```
