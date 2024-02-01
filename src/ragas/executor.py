import asyncio
import typing as t
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
from tqdm.auto import tqdm


@dataclass
class Executor:
    desc: str = "Evaluating"
    keep_progress_bar: bool = True
    futures: t.List[t.Any] = field(default_factory=list, repr=False)
    raise_exceptions: bool = False
    _is_new_eventloop: bool = False

    def __post_init__(self):
        try:
            self.executor = asyncio.get_running_loop()
        except RuntimeError:
            self.executor = asyncio.new_event_loop()
            self._is_new_eventloop = True

    def wrap_callable_with_index(self, callable: t.Callable, counter):
        async def wrapped_callable_async(*args, **kwargs):
            return counter, await callable(*args, **kwargs)

        return wrapped_callable_async

    def submit(
        self, callable: t.Callable, *args, name: t.Optional[str] = None, **kwargs
    ):
        self.executor = t.cast(asyncio.AbstractEventLoop, self.executor)
        callable_with_index = self.wrap_callable_with_index(callable, len(self.futures))
        # is type correct?
        callable_with_index = t.cast(t.Callable, callable_with_index)
        self.futures.append(
            self.executor.create_task(callable_with_index(*args, **kwargs), name=name)
        )

    async def _aresults(self) -> t.List[t.Any]:
        results = []
        for future in tqdm(
            asyncio.as_completed(self.futures),
            desc=self.desc,
            total=len(self.futures),
            # whether you want to keep the progress bar after completion
            leave=self.keep_progress_bar,
        ):
            r = (-1, np.nan)
            try:
                r = await future
            except Exception as e:
                if self.raise_exceptions:
                    raise e
            results.append(r)

        return results

    def results(self) -> t.List[t.Any]:
        results = []
        self.executor = t.cast(asyncio.AbstractEventLoop, self.executor)
        try:
            if self._is_new_eventloop:
                results = self.executor.run_until_complete(self._aresults())
            else:
                results = self.executor.create_task(self._aresults())
        finally:
            [f.cancel() for f in self.futures]
        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]
