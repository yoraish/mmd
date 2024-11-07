# Adapted from https://github.com/brouberol/contexttimer
from time import perf_counter

import torch


class TimerCUDA(object):
    """ A timer as a context manager

    Wraps around a timer. A custom timer can be passed
    to the constructor. The default timer is timeit.default_timer.

    Note that the latter measures wall clock time, not CPU time!
    On Unix systems, it corresponds to time.time.
    On Windows systems, it corresponds to time.clock.

    Keyword arguments:
        output -- if True, print output after exiting context.
                  if callable, pass output to callable.
        format -- str.format string to be used for output; default "took {} seconds"
        prefix -- string to prepend (plus a space) to output
                  For convenience, if you only specify this, output defaults to True.
    """

    def __init__(self, timer=perf_counter,
                 output=None, fmt="took {:.3f} seconds", prefix="",
                 use_cuda_events=False
                 ):
        self.timer = timer
        self.output = output
        self.fmt = fmt
        self.prefix = prefix
        self.start_time = None
        self.end = None

        self.sync_cuda = True if torch.cuda.is_available() else False
        if use_cuda_events:
            assert self.sync_cuda, "CUDA must be available when using CUDA events"
        self.use_cuda_events = use_cuda_events
        self.factor_cuda_events = 1./1000.  # transform to seconds
        self.start_event = None
        self.end_event = None

    def __call__(self):
        """ Return the current time """
        if self.use_cuda_events:
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event) * self.factor_cuda_events
        else:
            if self.sync_cuda:
                torch.cuda.synchronize()
            return self.timer()

    def __enter__(self):
        """ Set the start time """
        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = self()

        self.start = self()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Set the end time """
        self.end = self()

        if self.prefix and self.output is None:
            self.output = True

        if self.output:
            output = " ".join([self.prefix, self.fmt.format(self.elapsed)])
            if callable(self.output):
                self.output(output)
            else:
                print(output)

    def __str__(self):
        return f'{self.elapsed:.3f}'

    @property
    def elapsed(self):
        """ Return the current elapsed time since start

        If the `elapsed` property is called in the context manager scope,
        the elapsed time between start and property access is returned.
        However, if it is accessed outside of the context manager scope,
        it returns the elapsed time between entering and exiting the scope.

        The `elapsed` property can thus be accessed at different points within
        the context manager scope, to time different parts of the block.

        """
        if self.end is None:
            # if elapsed is called in the context manager scope
            if self.use_cuda_events:
                self()
            else:
                return self() - self.start
        else:
            # if elapsed is called out of the context manager scope
            if self.use_cuda_events:
                return self.end
            else:
                return self.end - self.start

