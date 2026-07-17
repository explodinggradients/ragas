import logging
import typing as t

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, Generation, LLMResult
from pydantic import BaseModel

from ragas.utils import get_from_dict

TokenUsageParser = t.Callable[[t.Union[LLMResult, ChatResult]], "TokenUsage"]

logger = logging.getLogger(__name__)


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    model: str = ""

    def __add__(self, y: "TokenUsage") -> "TokenUsage":
        if self.model == y.model or (self.model is None and y.model is None):
            return TokenUsage(
                input_tokens=self.input_tokens + y.input_tokens,
                output_tokens=self.output_tokens + y.output_tokens,
                model=self.model,
            )
        else:
            raise ValueError("Cannot add TokenUsage objects with different models")

    def cost(
        self,
        cost_per_input_token: float,
        cost_per_output_token: t.Optional[float] = None,
    ) -> float:
        if cost_per_output_token is None:
            cost_per_output_token = cost_per_input_token

        return (
            self.input_tokens * cost_per_input_token
            + self.output_tokens * cost_per_output_token
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenUsage):
            return False
        return (
            self.input_tokens == other.input_tokens
            and self.output_tokens == other.output_tokens
            and self.is_same_model(other)
        )

    def is_same_model(self, other: "TokenUsage") -> bool:
        if self.model is None and other.model is None:
            return True
        elif self.model == other.model:
            return True
        else:
            return False


def get_token_usage_for_openai(
    llm_result: t.Union[LLMResult, ChatResult],
) -> TokenUsage:
    # OpenAI like interfaces
    llm_output = llm_result.llm_output
    if llm_output is None:
        logger.info("No llm_output found in the LLMResult")
        return TokenUsage(input_tokens=0, output_tokens=0)
    output_tokens = get_from_dict(llm_output, "token_usage.completion_tokens", 0)
    input_tokens = get_from_dict(llm_output, "token_usage.prompt_tokens", 0)
    model = get_from_dict(llm_output, "model_name", "")

    return TokenUsage(
        input_tokens=input_tokens, output_tokens=output_tokens, model=model
    )


def _legacy_text_usage(
    llm_result: t.Union[LLMResult, ChatResult],
    *,
    usage_keys: t.Tuple[str, str],
    model_keys: t.Tuple[str, ...],
) -> TokenUsage:
    """Read token usage from legacy text-completion `Generation` entries.

    Some LangChain `BaseLLM` integrations expose the result as a `Generation`
    (not `ChatGeneration`) with usage living in ``generation_info`` or
    ``llm_output["usage"]``. This helper walks every entry in
    ``llm_result.generations``, picks the first non-zero match, and falls back
    to ``llm_output["usage"]`` for the same counts when needed.

    Returns ``TokenUsage(0, 0)`` if nothing usable is present.
    """
    for gs in llm_result.generations:
        for g in gs:
            if isinstance(g, ChatGeneration):
                continue
            if not isinstance(g, Generation):
                continue
            gen_info = g.generation_info or {}
            input_tokens = int(gen_info.get(usage_keys[0], 0) or 0)
            output_tokens = int(gen_info.get(usage_keys[1], 0) or 0)
            model = next(
                (gen_info.get(k, "") for k in model_keys if gen_info.get(k)), ""
            )
            if input_tokens or output_tokens:
                return TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                )
    llm_output = llm_result.llm_output or {}
    if not isinstance(llm_output, dict):
        return TokenUsage(input_tokens=0, output_tokens=0)
    usage = llm_output.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = int(usage.get(usage_keys[0], 0) or 0)
    output_tokens = int(usage.get(usage_keys[1], 0) or 0)
    if not (input_tokens or output_tokens):
        return TokenUsage(input_tokens=0, output_tokens=0)
    model = next((llm_output.get(k, "") for k in model_keys if llm_output.get(k)), "")
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )


def get_token_usage_for_anthropic(
    llm_result: t.Union[LLMResult, ChatResult],
) -> TokenUsage:
    token_usages = []
    for gs in llm_result.generations:
        for g in gs:
            if not isinstance(g, ChatGeneration):
                continue
            message: AIMessage = t.cast(AIMessage, g.message)
            usage_meta = message.usage_metadata or {}
            if not isinstance(usage_meta, dict):
                usage_meta = {}
            um_input = int(usage_meta.get("input_tokens", 0) or 0)
            um_output = int(usage_meta.get("output_tokens", 0) or 0)
            response_meta = message.response_metadata or {}
            if not isinstance(response_meta, dict):
                response_meta = {}
            r_input = int(get_from_dict(response_meta, "usage.input_tokens", 0) or 0)
            r_output = int(get_from_dict(response_meta, "usage.output_tokens", 0) or 0)
            # Prefer the LangChain-standard usage_metadata when populated; fall
            # back to response_metadata["usage.*"] for legacy chat integrations.
            input_tokens = um_input or r_input
            output_tokens = um_output or r_output
            if not (input_tokens or output_tokens):
                continue
            model = get_from_dict(response_meta, "model", "") or get_from_dict(
                response_meta, "model_name", ""
            )
            token_usages.append(
                TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                )
            )
    if token_usages:
        model = next((u.model for u in token_usages if u.model), "")
        return sum(
            token_usages, TokenUsage(input_tokens=0, output_tokens=0, model=model)
        )
    return _legacy_text_usage(
        llm_result,
        usage_keys=("input_tokens", "output_tokens"),
        model_keys=("model", "model_name", "model_id"),
    )


def get_token_usage_for_bedrock(
    llm_result: t.Union[LLMResult, ChatResult],
) -> TokenUsage:
    token_usages = []
    for gs in llm_result.generations:
        for g in gs:
            if not isinstance(g, ChatGeneration):
                continue
            message: AIMessage = t.cast(AIMessage, g.message)
            usage_meta = message.usage_metadata or {}
            if not isinstance(usage_meta, dict):
                usage_meta = {}
            um_input = int(usage_meta.get("input_tokens", 0) or 0)
            um_output = int(usage_meta.get("output_tokens", 0) or 0)
            response_meta = message.response_metadata or {}
            if not isinstance(response_meta, dict):
                response_meta = {}
            r_input = int(get_from_dict(response_meta, "usage.prompt_tokens", 0) or 0)
            r_output = int(
                get_from_dict(response_meta, "usage.completion_tokens", 0) or 0
            )
            input_tokens = um_input or r_input
            output_tokens = um_output or r_output
            if not (input_tokens or output_tokens):
                continue
            model = get_from_dict(response_meta, "model_id", "") or get_from_dict(
                response_meta, "model_name", ""
            )
            token_usages.append(
                TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                )
            )
    if token_usages:
        model = next((u.model for u in token_usages if u.model), "")
        return sum(
            token_usages, TokenUsage(input_tokens=0, output_tokens=0, model=model)
        )
    return _legacy_text_usage(
        llm_result,
        usage_keys=("prompt_tokens", "completion_tokens"),
        model_keys=("model_id", "model_name", "model"),
    )


def get_token_usage_for_azure_ai(
    llm_result: t.Union[LLMResult, ChatResult],
) -> TokenUsage:
    # AzureAI like interfaces
    llm_output = llm_result.llm_output
    if llm_output is None:
        logger.info("No llm_output found in the LLMResult")
        return TokenUsage(input_tokens=0, output_tokens=0)
    input_tokens = get_from_dict(llm_output, "token_usage.input_tokens", 0)
    output_tokens = get_from_dict(llm_output, "token_usage.output_tokens", 0)
    model = get_from_dict(llm_output, "model_name", "")

    return TokenUsage(
        input_tokens=input_tokens, output_tokens=output_tokens, model=model
    )


class CostCallbackHandler(BaseCallbackHandler):
    def __init__(self, token_usage_parser: TokenUsageParser):
        self.token_usage_parser = token_usage_parser
        self.usage_data: t.List[TokenUsage] = []

    def on_llm_end(self, response: LLMResult, **kwargs: t.Any):
        self.usage_data.append(self.token_usage_parser(response))

    def total_cost(
        self,
        cost_per_input_token: t.Optional[float] = None,
        cost_per_output_token: t.Optional[float] = None,
        per_model_costs: t.Dict[str, t.Tuple[float, float]] = {},
    ) -> float:
        if (
            per_model_costs == {}
            and cost_per_input_token is None
            and cost_per_output_token is None
        ):
            raise ValueError(
                "No cost table or cost per token provided. Please provide a cost table if using multiple models or cost per token if using a single model"
            )

        # sum up everything
        first_usage = self.usage_data[0]
        total_table: t.Dict[str, TokenUsage] = {first_usage.model: first_usage}
        for usage in self.usage_data[1:]:
            if usage.model in total_table:
                total_table[usage.model] += usage
            else:
                total_table[usage.model] = usage

        # caculate total cost
        # if only one model is used
        if len(total_table) == 1:
            model_name = list(total_table)[0]
            # if per model cost is provided check that
            if per_model_costs != {}:
                if model_name not in per_model_costs:
                    raise ValueError(f"Model {model_name} not found in per_model_costs")
                cpit, cpot = per_model_costs[model_name]
                return total_table[model_name].cost(cpit, cpot)
            # else use the cost_per_token vals
            else:
                if cost_per_output_token is None:
                    cost_per_output_token = cost_per_input_token
                assert cost_per_input_token is not None
                return total_table[model_name].cost(
                    cost_per_input_token, cost_per_output_token
                )
        else:
            total_cost = 0.0
            for model, usage in total_table.items():
                if model in per_model_costs:
                    cpit, cpot = per_model_costs[model]
                    total_cost += usage.cost(cpit, cpot)
            return total_cost

    def total_tokens(self) -> t.Union[TokenUsage, t.List[TokenUsage]]:
        """
        Return the sum of tokens used by the callback handler
        """
        first_usage = self.usage_data[0]
        total_table: t.Dict[str, TokenUsage] = {first_usage.model: first_usage}
        for usage in self.usage_data[1:]:
            if usage.model in total_table:
                total_table[usage.model] += usage
            else:
                total_table[usage.model] = usage

        if len(total_table) == 1:
            return list(total_table.values())[0]
        else:
            return list(total_table.values())
