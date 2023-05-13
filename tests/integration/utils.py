from __future__ import annotations

import time

import numpy as np


def timeit(func, iteration=3):
    def function_timer(*args, **kwargs) -> tuple(np.floating, np.floating):
        """
        Time the execution of a function and returns the time taken
        """
        # warmup
        func(*args, **kwargs)

        runtimes = []
        for _ in range(iteration):
            start = time.time()
            # we dont care about the return value
            func(*args, **kwargs)
            end = time.time()
            runtime = end - start
            runtimes.append(runtime)

        return np.mean(runtimes), np.var(runtimes)

    return function_timer
