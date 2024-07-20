import typing as t

from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainGroup,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainGroup,
    CallbackManagerForChainRun,
    Callbacks,
)


def new_group(
    name: str, inputs: t.Dict, callbacks: Callbacks
) -> t.Tuple[CallbackManagerForChainRun, CallbackManagerForChainGroup]:
    # start evaluation chain
    if isinstance(callbacks, list):
        cm = CallbackManager.configure(inheritable_callbacks=callbacks)
    else:
        cm = t.cast(CallbackManager, callbacks)
    rm = cm.on_chain_start({"name": name}, inputs)
    child_cm = rm.get_child()
    group_cm = CallbackManagerForChainGroup(
        child_cm.handlers,
        child_cm.inheritable_handlers,
        child_cm.parent_run_id,
        parent_run_manager=rm,
        tags=child_cm.tags,
        inheritable_tags=child_cm.inheritable_tags,
        metadata=child_cm.metadata,
        inheritable_metadata=child_cm.inheritable_metadata,
    )

    return rm, group_cm


async def new_async_group(
    name: str, inputs: t.Dict, callbacks: Callbacks
) -> t.Tuple[AsyncCallbackManagerForChainRun, AsyncCallbackManagerForChainGroup]:
    # start evaluation chain
    if isinstance(callbacks, list):
        cm = AsyncCallbackManager.configure(inheritable_callbacks=callbacks)
    else:
        cm = t.cast(AsyncCallbackManager, callbacks)
    rm = await cm.on_chain_start({"name": name}, inputs)
    child_cm = rm.get_child()
    group_cm = AsyncCallbackManagerForChainGroup(
        child_cm.handlers,
        child_cm.inheritable_handlers,
        child_cm.parent_run_id,
        parent_run_manager=rm,
        tags=child_cm.tags,
        inheritable_tags=child_cm.inheritable_tags,
        metadata=child_cm.metadata,
        inheritable_metadata=child_cm.inheritable_metadata,
    )
    return rm, group_cm
