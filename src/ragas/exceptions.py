from __future__ import annotations


class RagasException(Exception):
    """
    Base exception class for ragas.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ExceptionInRunner(RagasException):
    """
    Exception raised when an exception is raised in the executor.
    """

    def __init__(self):
        msg = "The runner thread which was running the jobs raised an exeception. Read the traceback above to debug it. You can also pass `raise_exceptions=False` incase you want to show only a warning message instead."
        super().__init__(msg)


class RagasOutputParserException(RagasException):
    """
    Exception raised when the output parser fails to parse the output.
    """

    def __init__(self, details: str = None):
        base_msg = "The output parser failed to parse the output including retries."
        if details:
            msg = f"{base_msg} Details: {details}"
        else:
            msg = base_msg
            
        # Add suggestions for local LLMs
        msg += "\nFor local LLMs, consider the following:\n" \
               "1. Increase the timeout in RunConfig (default is now 300 seconds)\n" \
               "2. Use a more capable local model that can better follow JSON formatting instructions\n" \
               "3. Reduce batch size to process fewer examples at once\n" \
               "4. For metrics that require structured output (context_recall, faithfulness, context_precision), " \
               "consider using simpler metrics like answer_correctness and answer_similarity if the issues persist"
        
        super().__init__(msg)


class LLMDidNotFinishException(RagasException):
    """
    Exception raised when the LLM did not finish.
    """

    def __init__(self):
        msg = "The LLM generation was not completed. Please increase the max_tokens and try again."
        super().__init__(msg)


class UploadException(RagasException):
    """
    Exception raised when the app fails to upload the results.
    """

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)
