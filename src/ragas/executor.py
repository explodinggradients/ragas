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
    is_async: bool = True
    max_workers: t.Optional[int] = None
    futures: t.List[t.Any] = field(default_factory=list, repr=False)
    raise_exceptions: bool = False
    _is_new_eventloop: bool = False

    def __post_init__(self):
        if self.is_async:
            try:
                self.executor = asyncio.get_running_loop()
            except RuntimeError:
                self.executor = asyncio.new_event_loop()
                self._is_new_eventloop = True
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def _validation_for_mode(self):
        if self.is_async and self.max_workers is not None:
            raise ValueError(
                "Cannot evaluate with both async and threads. Either set is_async=False or max_workers=None."  # noqa
            )

    def wrap_callable_with_index(self, callable: t.Callable, counter):
        def wrapped_callable(*args, **kwargs):
            return counter, callable(*args, **kwargs)

        async def wrapped_callable_async(*args, **kwargs):
            return counter, await callable(*args, **kwargs)

        if self.is_async:
            return wrapped_callable_async
        else:
            return wrapped_callable

    def submit(
        self, callable: t.Callable, *args, name: t.Optional[str] = None, **kwargs
    ):
        if self.is_async:
            self.executor = t.cast(asyncio.AbstractEventLoop, self.executor)
            callable_with_index = self.wrap_callable_with_index(
                callable, len(self.futures)
            )
            # is type correct?
            callable_with_index = t.cast(t.Callable, callable_with_index)
            self.futures.append(
                self.executor.create_task(
                    callable_with_index(*args, **kwargs), name=name
                )
            )
        else:
            self.executor = t.cast(ThreadPoolExecutor, self.executor)
            callable_with_index = self.wrap_callable_with_index(
                callable, len(self.futures)
            )
            self.futures.append(
                self.executor.submit(callable_with_index, *args, **kwargs)
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
        if self.is_async:
            self.executor = t.cast(asyncio.AbstractEventLoop, self.executor)
            try:
                if self._is_new_eventloop:
                    results = self.executor.run_until_complete(self._aresults())

                # event loop is running use nested_asyncio to hijack the event loop
                else:
                    import nest_asyncio

                    nest_asyncio.apply()
                    results = self.executor.run_until_complete(self._aresults())
            finally:
                [f.cancel() for f in self.futures]

        else:
            self.executor = t.cast(ThreadPoolExecutor, self.executor)
            try:
                for future in tqdm(
                    as_completed(self.futures),
                    desc=self.desc,
                    total=len(self.futures),
                    # whether you want to keep the progress bar after completion
                    leave=self.keep_progress_bar,
                ):
                    r = (-1, np.nan)
                    try:
                        r = future.result()
                    except Exception as e:
                        r = (-1, np.nan)
                        if self.raise_exceptions:
                            raise e
                    finally:
                        results.append(r)
            finally:
                self.executor.shutdown(wait=False)

        sorted_results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]
