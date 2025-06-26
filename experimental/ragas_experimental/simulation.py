"""
User Simulator for multi-turn conversation evaluation.

This module provides functionality to simulate realistic user interactions
for evaluating conversational AI systems.
"""

import inspect
import re
import typing as t

from pydantic import BaseModel, Field

from .llm.llm import RagasLLM
from .prompt.base import Prompt


class Message(BaseModel):
    """Represents a single message in a conversation."""

    role: t.Literal["user", "assistant"]
    content: t.Union[str, t.Dict[str, t.Any], t.List[t.Dict[str, t.Any]]]


class ConversationHistory(BaseModel):
    """Represents the full conversation history."""

    messages: t.List[Message] = Field(default_factory=list)

    def add_message(
        self,
        role: t.Literal["user", "assistant"],
        content: t.Union[str, t.Dict[str, t.Any], t.List[t.Dict[str, t.Any]]],
    ) -> None:
        """Add a message to the conversation history."""
        self.messages.append(Message(role=role, content=content))

    def get_last_message(self) -> t.Optional[Message]:
        """Get the last message in the conversation."""
        return self.messages[-1] if self.messages else None

    def to_dict_list(self) -> t.List[t.Dict[str, t.Any]]:
        """Convert conversation history to a list of dictionaries."""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]


def validate_agent_function(func: t.Callable) -> None:
    """
    Validate agent function signature and behavior.

    Checks:
    1. Function accepts at least 2 parameters (query, history)
    2. Function can handle basic inputs without TypeError
    3. Function returns something (not None)

    Supports flexible agent signatures for multimodal agents:
    - Input: text, images, mixed content
    - Output: str, dict with 'content' key, or any serializable type
    """
    # 1. Signature validation
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) < 2:
        raise ValueError(
            f"Agent function must accept at least 2 parameters (query, history), got {len(params)}"
        )

    # 2. Test call with mock data
    try:
        mock_history = ConversationHistory()
        mock_history.add_message("user", "test query")

        result = func("test query", mock_history)

        # 3. Return type validation - just ensure it's not None
        if result is None:
            raise ValueError("Agent function cannot return None")

    except TypeError as e:
        raise ValueError(f"Agent function signature invalid: {e}")


def validate_stopping_criteria(func: t.Callable[[ConversationHistory], bool]) -> None:
    """
    Validate stopping criteria function signature and behavior.

    Checks:
    1. Function accepts exactly 1 parameter: (history: ConversationHistory)
    2. Function returns boolean or boolean-convertible value
    3. Function doesn't raise TypeError on valid ConversationHistory
    """
    # 1. Signature validation
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) != 1:
        raise ValueError(
            f"Stopping criteria must accept exactly 1 parameter (history), got {len(params)}"
        )

    # 2. Test call with mock data
    try:
        mock_history = ConversationHistory()
        mock_history.add_message("user", "test")
        mock_history.add_message("assistant", "response")

        result = func(mock_history)

        # 3. Return type validation
        if result is None:
            raise ValueError("Stopping criteria cannot return None")

        # Ensure it's boolean convertible
        bool(result)

    except TypeError as e:
        raise ValueError(f"Stopping criteria signature invalid: {e}")


class UserSimulatorResponse(BaseModel):
    """Response from the user simulator."""

    content: str = Field(description="The simulated user response")
    should_continue: bool = Field(
        default=True, description="Whether the conversation should continue"
    )


class UserSimulator:
    """
    Simulates realistic user interactions for conversational AI evaluation.

    This class can generate user responses based on personas, behaviors, and
    conversation context to create realistic multi-turn evaluations.
    """

    def __init__(
        self,
        prompt: Prompt,
        llm: RagasLLM,
        agent_function: t.Callable,
        stopping_criteria: t.Optional[t.Callable[[ConversationHistory], bool]] = None,
        max_turns: int = 10,
        **kwargs,
    ):
        """
        Initialize the UserSimulator.

        Args:
            prompt: The prompt template for generating user responses
            llm: The language model to use for generating responses
            agent_function: The agent function to interact with during simulation
            stopping_criteria: Optional function to determine when to stop the conversation
            max_turns: Maximum number of conversation turns (default: 10)
            **kwargs: Additional parameters for customization
        """
        # Check if conversation_history is already in the prompt, if not add it
        placeholders = re.findall(r"\{(\w+)\}", prompt.instruction)
        if "conversation_history" not in placeholders:
            # Add conversation_history to the prompt instruction
            prompt.instruction += "\n\nConversation History:\n{conversation_history}"

        self.prompt = prompt
        self.llm = llm
        self.agent_function = agent_function
        self.stopping_criteria = stopping_criteria or self._default_stopping_criteria
        self.max_turns = max_turns
        self.kwargs = kwargs

        # Validate agent function and stopping criteria
        validate_agent_function(self.agent_function)
        validate_stopping_criteria(self.stopping_criteria)

    def _default_stopping_criteria(
        self, conversation_history: ConversationHistory
    ) -> bool:
        """Default stopping criteria based on conversation length."""
        return len(conversation_history.messages) >= self.max_turns

    def _should_stop_conversation(
        self, conversation_history: ConversationHistory
    ) -> bool:
        """Check if the conversation should be stopped."""
        try:
            result = self.stopping_criteria(conversation_history)
            return bool(result)
        except Exception as e:
            # If stopping criteria fails, stop conversation to avoid infinite loop
            print(
                f"Warning: Stopping criteria failed with error: {e}. Stopping conversation."
            )
            return True

    def _generate_user_response(
        self, conversation_history: ConversationHistory, **context_vars
    ) -> UserSimulatorResponse:
        """
        Generate a user response based on conversation history and context.

        Args:
            conversation_history: The current conversation history
            **context_vars: Additional context variables for prompt formatting

        Returns:
            UserSimulatorResponse containing the generated response
        """
        # Prepare prompt variables including conversation_history
        prompt_vars = {
            **context_vars,
            **self.kwargs,
            "conversation_history": self._format_conversation_for_prompt(
                conversation_history
            ),
        }

        # Generate the prompt
        formatted_prompt = self.prompt.format(**prompt_vars)

        # Generate response using LLM
        response = self.llm.generate(formatted_prompt, UserSimulatorResponse)

        return response

    def _format_conversation_for_prompt(
        self, conversation_history: ConversationHistory
    ) -> str:
        """Format conversation history for inclusion in prompts."""
        if not conversation_history.messages:
            return "No previous conversation."

        formatted_messages = []
        for msg in conversation_history.messages:
            # Handle different content types
            if isinstance(msg.content, str):
                content_str = msg.content
            else:
                # Convert dict/list content to string representation
                content_str = str(msg.content)
            formatted_messages.append(f"{msg.role.title()}: {content_str}")

        return "\n".join(formatted_messages)

    def run(
        self, initial_message: t.Optional[t.Dict[str, str]] = None, **context_vars
    ) -> ConversationHistory:
        """
        Run a complete conversation simulation.

        Args:
            initial_message: Optional initial message to start the conversation
            **context_vars: Additional context variables for the simulation

        Returns:
            ConversationHistory containing the complete conversation
        """
        conversation_history = ConversationHistory()

        # Add initial message if provided
        if initial_message:
            role = initial_message.get("role", "user")
            content = initial_message.get("content", "")
            # Ensure role is valid
            if role not in ["user", "assistant"]:
                role = "user"
            conversation_history.add_message(
                t.cast(t.Literal["user", "assistant"], role), content
            )

        # Continue conversation until stopping criteria is met
        while not self._should_stop_conversation(conversation_history):
            last_message = conversation_history.get_last_message()

            # If last message was from user, get agent response
            if last_message and last_message.role == "user":
                try:
                    # Call the agent function with the conversation history
                    agent_response = self.agent_function(
                        last_message.content, conversation_history
                    )

                    # Add agent response to conversation
                    if isinstance(agent_response, str):
                        conversation_history.add_message("assistant", agent_response)
                    elif (
                        isinstance(agent_response, dict) and "content" in agent_response
                    ):
                        role = agent_response.get("role", "assistant")
                        if role not in ["user", "assistant"]:
                            role = "assistant"
                        conversation_history.add_message(
                            role, agent_response["content"]
                        )
                    else:
                        # Handle other response formats
                        conversation_history.add_message(
                            "assistant", str(agent_response)
                        )

                except Exception as e:
                    # Handle agent function errors gracefully
                    conversation_history.add_message("assistant", f"Error: {str(e)}")

            # If conversation should continue, generate user response
            if not self._should_stop_conversation(conversation_history):
                user_response = self._generate_user_response(
                    conversation_history, **context_vars
                )

                # Add user response to conversation
                conversation_history.add_message("user", user_response.content)

                # Check if user wants to stop
                if not user_response.should_continue:
                    break

        return conversation_history


def default_stopping_criteria(conversation_history: ConversationHistory) -> bool:
    """
    Default stopping criteria function.

    Stops conversation when it reaches 10 messages or more.
    """
    return len(conversation_history.messages) >= 10
