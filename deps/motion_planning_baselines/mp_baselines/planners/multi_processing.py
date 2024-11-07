import torch.multiprocessing as mp


class MultiProcessor:
    # Adapated from https://stackoverflow.com/a/45829852

    def __init__(self, max_processes=-1, start_method='forkserver', **kwargs):
        mp.set_start_method(start_method)
        self.pool = mp.Pool(processes=mp.cpu_count() if max_processes == -1 else max_processes)
        self.async_results = []

    @staticmethod
    def _wrapper(func,  args, kwargs):
        ret = func(*args, **kwargs)
        return ret

    def run(self, func, *args, **kwargs):
        args2 = [func, args, kwargs]
        async_res = self.pool.apply_async(self._wrapper, args=args2)
        self.async_results.append(async_res)

    def wait(self):
        self.pool.close()
        self.pool.join()
        res_l = []
        for async_res in self.async_results:
            res = async_res.get()
            res_l.append(res)
        return res_l
