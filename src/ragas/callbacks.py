from __future__ import annotations

import json
import typing as t
import uuid
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.callbacks import (
    BaseCallbackHandler,
    CallbackManager,
    CallbackManagerForChainGroup,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


def new_group(
    name: str,
    inputs: t.Dict,
    callbacks: Callbacks,
    tags: t.List[str] | None = None,
    metadata: t.Dict[str, t.Any] | None = None,
) -> t.Tuple[CallbackManagerForChainRun, CallbackManagerForChainGroup]:
    tags = tags or []
    metadata = metadata or {}

    # start evaluation chain
    if isinstance(callbacks, list):
        cm = CallbackManager.configure(inheritable_callbacks=callbacks)
    else:
        cm = t.cast(CallbackManager, callbacks)
    cm.tags = tags
    cm.metadata = metadata
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


class ChainType(Enum):
    EVALUATION = "evaluation"
    METRIC = "metric"
    ROW = "row"
    RAGAS_PROMPT = "ragas_prompt"


class ChainRuns(BaseModel):
    run_id: uuid.UUID
    parent_run_id: uuid.UUID | None
    name: str
    inputs: t.Dict[str, t.Any]
    metadata: t.Dict[str, t.Any]
    outputs: t.Dict[str, t.Any] = Field(default_factory=dict)
    children: t.List[uuid.UUID] = Field(default_factory=list)


class ChainRunEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, uuid.UUID):
            return str(o)
        if isinstance(o, ChainType):
            return o.value
        return json.JSONEncoder.default(self, o)


@dataclass
class RagasTracer(BaseCallbackHandler):
    traces: t.Dict[uuid.UUID, ChainRuns] = field(default_factory=dict)

    def on_chain_start(
        self,
        serialized: t.Dict[str, t.Any],
        inputs: t.Dict[str, t.Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: t.Optional[uuid.UUID] = None,
        tags: t.Optional[t.List[str]] = None,
        metadata: t.Optional[t.Dict[str, t.Any]] = None,
        **kwargs: t.Any,
    ) -> t.Any:
        self.traces[run_id] = ChainRuns(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=serialized["name"],
            inputs=inputs,
            metadata=metadata or {},
            children=[],
        )

        if parent_run_id and parent_run_id in self.traces:
            self.traces[parent_run_id].children.append(run_id)

    def on_chain_end(
        self,
        outputs: t.Dict[str, t.Any],
        *,
        run_id: uuid.UUID,
        **kwargs: t.Any,
    ) -> t.Any:
        self.traces[run_id].outputs = outputs

    def to_jsons(self) -> str:
        return json.dumps(
            [t.model_dump() for t in self.traces.values()],
            indent=4,
            cls=ChainRunEncoder,
        )
