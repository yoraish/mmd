The original repository is this one: https://github.com/anindex/motion_planning_baselines.
Their original README follows.

# Motion Planning Baselines in PyTorch

This library implements various Motion Planning methods.

**NOTE**: `mp_baslines` is under heavy development and highly experimental.

## Installation

Simply activate your conda/Python environment and run

```bash
pip install -e .
```

## Examples

```bash
python examples/pointmass_dense_2d_CHOMP.py
python examples/pointmass_dense_2d_GPMP.py
python examples/pointmass_grid_circles_2d_Stoch-GPMP.py
python examples/pointmass_grid_circles_2d_STOMP.py
python examples/pointmass_grid_circles_2d_MPPI.py
python examples/pointmass_dense_2d_RRT_multiprocess.py
```

## Contact

If you have any questions or find any bugs, please let us know:

- [An Le](https://www.ias.informatik.tu-darmstadt.de/Team/AnThaiLe), [an@robot-learning.de](an@robot-learning.de)
- [Joao Carvalho](https://www.ias.informatik.tu-darmstadt.de/Team/JoaoCarvalho), [joao@robot-learning.de](joao@robot-learning.de)

## Citation

If you found this repository useful, please consider citing these references:

```bibtex
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
