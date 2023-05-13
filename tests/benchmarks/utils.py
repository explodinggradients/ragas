from __future__ import annotations

import time

import numpy as np
from rich.console import Console
from rich.table import Table


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


def print_table(result):
    table = Table("Batch Name", "(mean, var)", title="Benchmark Results")

    for batch_name, (mean, var) in result.items():
        table.add_row(batch_name, f"{mean:.4f}, {var:.4f}")

    console = Console()
    console.print(table)
