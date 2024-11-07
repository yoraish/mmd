import random

import numpy as np
import torch


def fix_random_seed(seed):
    random.seed(seed)

    try:
        np.random.seed(seed)
    except NameError:
        pass

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
        torch.backends.cudnn.benchmark = False
    except NameError:
        pass
