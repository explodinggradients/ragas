import asyncio
import typing as t
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from tqdm.auto import tqdm


@dataclass
class Executor:
    in_async_mode: bool = True
    max_workers: t.Optional[int] = None
    futures: t.List[t.Any] = field(default_factory=list, repr=False)
    raise_exceptions: bool = False

    def __post_init__(self):
        if self.in_async_mode:
            self.executor = asyncio.get_event_loop()
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def _validation_for_mode(self):
        if self.in_async_mode and self.max_workers is not None:
            raise ValueError(
                "Cannot evaluate with both async and threads. Either set is_async=False or max_workers=None."  # noqa
            )

    def submit(self, callable: t.Callable, *args, **kwargs):
        if self.in_async_mode:
            self.executor = t.cast(asyncio.AbstractEventLoop, self.executor)
            self.futures.append(self.executor.create_task(callable(*args, **kwargs)))
        else:
            self.executor = t.cast(ThreadPoolExecutor, self.executor)
            self.futures.append(self.executor.submit(callable, *args, **kwargs))

    async def aresults(self) -> t.List[t.Any]:
        results = []
        for future in tqdm(self.futures, desc="Evaluating"):
            results.append(await future)

        return results

    def results(self) -> t.List[t.Any]:
        results = []
        for future in tqdm(self.futures, desc="Evaluating"):
            results.append(future.result())
        return results
