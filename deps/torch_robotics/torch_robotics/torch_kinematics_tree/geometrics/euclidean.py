
def e_log_map(x, base=None):
    x = x.unsqueeze(0) if len(x.shape) == 1 else x
    if base is None:
        return x
    else:
        return x - base


def e_exp_map(x, base=None):
    x = x.unsqueeze(0) if len(x.shape) == 1 else x
    if base is None:
        return x
    else:
        return x + base


def e_parallel_transport(x, g, h):
    x = x.unsqueeze(0) if len(x.shape) == 1 else x
    return x