import json
import re
import warnings

from langchain.schema.document import Document as LangchainDocument
from llama_index.readers.schema import Document as LlamaindexDocument


def langchain_to_llamaindex_documents(
    langchain_docs: list[LangchainDocument],
) -> list[LlamaindexDocument]:
    ...


def load_as_json(text):
    """
    validate and return given text as json
    """

    try:
        return json.loads(text)
    except ValueError:
        warnings.warn("Invalid json")

    return {}


def load_as_score(text):
    """
    validate and returns given text as score
    """

    pattern = r"^[\d.]+$"
    if not re.match(pattern, text):
        warnings.warn("Invalid score")
        score = 0.0
    else:
        score = eval(text)

    return score
