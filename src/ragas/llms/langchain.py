from __future__ import annotations

import typing as t

from langchain.chat_models import AzureChatOpenAI, BedrockChat, ChatOpenAI, ChatVertexAI
from langchain.chat_models.base import BaseChatModel
from langchain.llms import AmazonAPIGateway, AzureOpenAI, Bedrock, OpenAI, VertexAI
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult

from ragas.async_utils import run_async_tasks
from ragas.exceptions import AzureOpenAIKeyNotFound, OpenAIKeyNotFound
from ragas.llms.base import RagasLLM
from ragas.utils import NO_KEY

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks
    from langchain.prompts import ChatPromptTemplate


def isOpenAI(llm: BaseLLM | BaseChatModel) -> bool:
    return isinstance(llm, OpenAI) or isinstance(llm, ChatOpenAI)


def isBedrock(llm: BaseLLM | BaseChatModel) -> bool:
    return isinstance(llm, Bedrock) or isinstance(llm, BedrockChat)

def isAmazonAPIGateway(llm: BaseLLM | BaseChatModel) -> bool:
    return isinstance(llm, AmazonAPIGateway)


# have to specify it twice for runtime and static checks
MULTIPLE_COMPLETION_SUPPORTED = [
    OpenAI,
    ChatOpenAI,
    AzureOpenAI,
    AzureChatOpenAI,
    ChatVertexAI,
    VertexAI,
]
MultipleCompletionSupportedLLM = t.Union[
    OpenAI, ChatOpenAI, AzureOpenAI, AzureChatOpenAI, ChatVertexAI, VertexAI
]


def _compute_token_usage_langchain(list_llmresults: t.List[LLMResult]) -> t.Dict:
    # compute total token usage by adding individual token usage
    llm_output = list_llmresults[0].llm_output
    if llm_output is None:
        return {}
    if (llm_output is not None) and ("token_usage" in llm_output):
        sum_prompt_tokens = 0
        sum_completion_tokens = 0
        sum_total_tokens = 0
        for result in list_llmresults:
            if result.llm_output is None:
                continue
            token_usage = result.llm_output["token_usage"]
            sum_prompt_tokens += token_usage["prompt_tokens"]
            sum_completion_tokens += token_usage["completion_tokens"]
            sum_total_tokens += token_usage["total_tokens"]

        llm_output["token_usage"] = {
            "prompt_tokens": sum_prompt_tokens,
            "completion_tokens": sum_completion_tokens,
            "sum_total_tokens": sum_total_tokens,
        }

    return llm_output


class LangchainLLM(RagasLLM):
    n_completions_supported: bool = True

    def __init__(self, llm: BaseLLM | BaseChatModel):
        self.langchain_llm = llm

    @property
    def llm(self):
        return self.langchain_llm

    def validate_api_key(self):
        # if langchain OpenAI or ChatOpenAI
        if isinstance(self.llm, ChatOpenAI) or isinstance(self.llm, OpenAI):
            # make sure the type is LangchainLLM with ChatOpenAI
            self.langchain_llm = t.cast(ChatOpenAI, self.langchain_llm)
            # raise error if no api key
            if self.langchain_llm.openai_api_key == NO_KEY:
                raise OpenAIKeyNotFound

        # if langchain AzureOpenAI or ChatAzurerOpenAI
        elif isinstance(self.llm, AzureChatOpenAI) or isinstance(self.llm, AzureOpenAI):
            self.langchain_llm = t.cast(AzureChatOpenAI, self.langchain_llm)
            # raise error if no api key
            if self.langchain_llm.openai_api_key == NO_KEY:
                raise AzureOpenAIKeyNotFound

    @staticmethod
    def llm_supports_completions(llm):
        for llm_type in MULTIPLE_COMPLETION_SUPPORTED:
            if isinstance(llm, llm_type):
                return True

    def _generate_multiple_completions(
        self,
        prompts: list[ChatPromptTemplate],
        n: int = 1,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        self.langchain_llm = t.cast(MultipleCompletionSupportedLLM, self.langchain_llm)
        old_n = self.langchain_llm.n
        self.langchain_llm.n = n

        if isinstance(self.llm, BaseLLM):
            ps = [p.format() for p in prompts]
            result = self.llm.generate(ps, callbacks=callbacks)
        else:  # if BaseChatModel
            ps = [p.format_messages() for p in prompts]
            result = self.llm.generate(ps, callbacks=callbacks)
        self.llm.n = old_n

        return result

    async def generate_completions(
        self,
        prompts: list[ChatPromptTemplate],
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        if isinstance(self.llm, BaseLLM):
            ps = [p.format() for p in prompts]
            result = await self.llm.agenerate(ps, callbacks=callbacks)
        else:  # if BaseChatModel
            ps = [p.format_messages() for p in prompts]
            result = await self.llm.agenerate(ps, callbacks=callbacks)

        return result

    async def agenerate(
        self,
        prompt: ChatPromptTemplate,
        n: int = 1,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        temperature = 0.2 if n > 1 else 0
        if isBedrock(self.llm) and ("model_kwargs" in self.llm.__dict__):
            self.llm.model_kwargs = {"temperature": temperature}
        else:
            self.llm.temperature = temperature

        if self.llm_supports_completions(self.llm):
            self.langchain_llm = t.cast(
                MultipleCompletionSupportedLLM, self.langchain_llm
            )
            old_n = self.langchain_llm.n
            self.langchain_llm.n = n
            if isinstance(self.llm, BaseLLM):
                result = await self.llm.agenerate(
                    [prompt.format()], callbacks=callbacks
                )
            else:  # if BaseChatModel
                result = await self.llm.agenerate(
                    [prompt.format_messages()], callbacks=callbacks
                )
            self.langchain_llm.n = old_n
        else:
            if isinstance(self.llm, BaseLLM):
                list_llmresults: list[LLMResult] = run_async_tasks(
                    [
                        self.llm.agenerate([prompt.format()], callbacks=callbacks)
                        for _ in range(n)
                    ]
                )
            else:
                list_llmresults: list[LLMResult] = run_async_tasks(
                    [
                        self.llm.agenerate(
                            [prompt.format_messages()], callbacks=callbacks
                        )
                        for _ in range(n)
                    ]
                )

            # fill results as if the LLM supported multiple completions
            generations = [r.generations[0][0] for r in list_llmresults]
            llm_output = _compute_token_usage_langchain(list_llmresults)
            result = LLMResult(generations=[generations], llm_output=llm_output)

        return result

    def generate(
        self,
        prompts: list[ChatPromptTemplate],
        n: int = 1,
        temperature: float = 1e-8,
        callbacks: t.Optional[Callbacks] = None,
    ) -> LLMResult:
        # set temperature to 0.2 for multiple completions
        temperature = 0.2 if n > 1 else 1e-8
        if isBedrock(self.llm) and ("model_kwargs" in self.llm.__dict__):
            self.llm.model_kwargs = {"temperature": temperature}
        elif isAmazonAPIGateway(self.llm) and ("model_kwargs" in self.llm.__dict__):
            self.llm.model_kwargs = {"temperature": temperature}
        else:
            self.llm.temperature = temperature

        if self.llm_supports_completions(self.llm):
            return self._generate_multiple_completions(prompts, n, callbacks)
        else:  # call generate_completions n times to mimic multiple completions
            list_llmresults = run_async_tasks(
                [self.generate_completions(prompts, callbacks) for _ in range(n)]
            )

            # fill results as if the LLM supported multiple completions
            generations = []
            for i in range(len(prompts)):
                completions = []
                for result in list_llmresults:
                    completions.append(result.generations[i][0])
                generations.append(completions)

            llm_output = _compute_token_usage_langchain(list_llmresults)
            return LLMResult(generations=generations, llm_output=llm_output)
