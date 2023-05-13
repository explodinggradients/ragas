from __future__ import annotations

import time
import typing as t

import numpy as np
from rich.console import Console
from rich.table import Table

P = t.ParamSpec("P")
R = t.TypeVar("R")
OrigFunc = t.Callable[P, R]
DecoratedFunc = t.Callable[P, tuple[np.floating, np.floating]]


def timeit(func: OrigFunc, iteration: int = 3) -> DecoratedFunc:
    def function_timer(
        *args: P.args, **kwargs: P.kwargs
    ) -> tuple[np.floating, np.floating]:
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
