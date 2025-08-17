"""
Tests for the simulation module.

Comprehensive test suite covering Message, ConversationHistory, UserSimulator,
validation functions, and conversation flow scenarios.
"""

import pytest
from unittest.mock import Mock
from pydantic import ValidationError

from ragas.simulation import (
    Message,
    ConversationHistory,
    UserSimulator,
    UserSimulatorResponse,
    validate_agent_function,
    validate_stopping_criteria,
    default_stopping_criteria,
)
from ragas.prompt.base import Prompt


class TestMessage:
    """Test Message class functionality."""

    def test_message_creation_valid(self):
        """Test creating valid Message instances."""
        # String content
        msg1 = Message(role="user", content="Hello world")
        assert msg1.role == "user"
        assert msg1.content == "Hello world"

        # Dict content
        msg2 = Message(role="assistant", content={"text": "response", "type": "chat"})
        assert msg2.role == "assistant"
        assert isinstance(msg2.content, dict)

        # List content
        msg3 = Message(role="user", content=[{"type": "text", "content": "hello"}])
        assert msg3.role == "user"
        assert isinstance(msg3.content, list)

    def test_message_invalid_role(self):
        """Test Message with invalid role raises validation error."""
        with pytest.raises(ValidationError):
            Message(role="invalid", content="test")


class TestConversationHistory:
    """Test ConversationHistory class functionality."""

    def test_empty_conversation_history(self):
        """Test creating empty conversation history."""
        history = ConversationHistory()
        assert len(history.messages) == 0
        assert history.get_last_message() is None

    def test_add_message(self):
        """Test adding messages to conversation history."""
        history = ConversationHistory()

        history.add_message("user", "First message")
        assert len(history.messages) == 1
        assert history.messages[0].role == "user"
        assert history.messages[0].content == "First message"

        history.add_message("assistant", "Response message")
        assert len(history.messages) == 2
        assert history.messages[1].role == "assistant"

    def test_get_last_message(self):
        """Test getting the last message from conversation history."""
        history = ConversationHistory()

        # Empty history
        assert history.get_last_message() is None

        # Single message
        history.add_message("user", "Hello")
        last = history.get_last_message()
        assert last is not None
        assert last.role == "user"
        assert last.content == "Hello"

        # Multiple messages
        history.add_message("assistant", "Hi there")
        last = history.get_last_message()
        assert last.role == "assistant"
        assert last.content == "Hi there"

    def test_to_dict_list(self):
        """Test converting conversation to dictionary list."""
        history = ConversationHistory()

        # Empty history
        dict_list = history.to_dict_list()
        assert dict_list == []

        # With messages
        history.add_message("user", "Question")
        history.add_message("assistant", {"text": "Answer"})

        dict_list = history.to_dict_list()
        assert len(dict_list) == 2
        assert dict_list[0] == {"role": "user", "content": "Question"}
        assert dict_list[1] == {"role": "assistant", "content": {"text": "Answer"}}


class TestValidationFunctions:
    """Test validation functions for agent and stopping criteria."""

    def test_validate_agent_function_valid(self):
        """Test validation of valid agent functions."""

        # Basic valid function
        def valid_agent(query, history):
            return "response"

        validate_agent_function(valid_agent)  # Should not raise

        # Function with extra parameters
        def valid_agent_extra(query, history, context=None):
            return "response"

        validate_agent_function(valid_agent_extra)  # Should not raise

    def test_validate_agent_function_invalid_params(self):
        """Test validation fails for functions with insufficient parameters."""

        # Too few parameters
        def invalid_agent(query):
            return "response"

        with pytest.raises(ValueError, match="must accept at least 2 parameters"):
            validate_agent_function(invalid_agent)

        # No parameters
        def invalid_agent_no_params():
            return "response"

        with pytest.raises(ValueError, match="must accept at least 2 parameters"):
            validate_agent_function(invalid_agent_no_params)

    def test_validate_agent_function_returns_none(self):
        """Test validation fails for functions returning None."""

        def invalid_agent(query, history):
            return None

        with pytest.raises(ValueError, match="cannot return None"):
            validate_agent_function(invalid_agent)

    def test_validate_agent_function_type_error(self):
        """Test validation fails for functions with signature issues."""

        def invalid_agent(query, history, required_param):
            return "response"

        with pytest.raises(ValueError, match="signature invalid"):
            validate_agent_function(invalid_agent)

    def test_validate_stopping_criteria_valid(self):
        """Test validation of valid stopping criteria functions."""

        def valid_stop(history):
            return len(history.messages) > 5

        validate_stopping_criteria(valid_stop)  # Should not raise

    def test_validate_stopping_criteria_invalid_params(self):
        """Test validation fails for stopping criteria with wrong parameters."""

        # Too many parameters
        def invalid_stop(history, extra):
            return True

        with pytest.raises(ValueError, match="exactly 1 parameter"):
            validate_stopping_criteria(invalid_stop)

        # No parameters
        def invalid_stop_no_params():
            return True

        with pytest.raises(ValueError, match="exactly 1 parameter"):
            validate_stopping_criteria(invalid_stop_no_params)

    def test_validate_stopping_criteria_returns_none(self):
        """Test validation fails for stopping criteria returning None."""

        def invalid_stop(history):
            return None

        with pytest.raises(ValueError, match="cannot return None"):
            validate_stopping_criteria(invalid_stop)

    def test_default_stopping_criteria(self):
        """Test default stopping criteria function."""
        history = ConversationHistory()

        # Empty history - should not stop
        assert not default_stopping_criteria(history)

        # Add messages up to threshold
        for i in range(9):
            history.add_message("user" if i % 2 == 0 else "assistant", f"message {i}")
        assert not default_stopping_criteria(history)

        # At threshold - should stop
        history.add_message("assistant", "final message")
        assert default_stopping_criteria(history)


class TestUserSimulatorResponse:
    """Test UserSimulatorResponse class."""

    def test_user_simulator_response_creation(self):
        """Test creating UserSimulatorResponse instances."""
        # Default values
        response = UserSimulatorResponse(content="Hello")
        assert response.content == "Hello"
        assert response.should_continue is True

        # Explicit values
        response2 = UserSimulatorResponse(content="Goodbye", should_continue=False)
        assert response2.content == "Goodbye"
        assert response2.should_continue is False


class TestUserSimulator:
    """Test UserSimulator class functionality."""

    @pytest.fixture
    def mock_prompt(self):
        """Create a mock prompt for testing."""
        return Prompt("You are a helpful user. Respond to: {query}")

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock()
        llm.generate.return_value = UserSimulatorResponse(content="Mock response")
        return llm

    @pytest.fixture
    def simple_agent(self):
        """Create a simple agent function for testing."""

        def agent(query, history):
            return f"Agent response to: {query}"

        return agent

    def test_user_simulator_initialization(self, mock_prompt, mock_llm, simple_agent):
        """Test UserSimulator initialization."""
        simulator = UserSimulator(
            prompt=mock_prompt, llm=mock_llm, agent_function=simple_agent, max_turns=5
        )

        assert simulator.prompt == mock_prompt
        assert simulator.llm == mock_llm
        assert simulator.agent_function == simple_agent
        assert simulator.max_turns == 5

    def test_user_simulator_prompt_modification(self, mock_llm, simple_agent):
        """Test that conversation_history is added to prompt if missing."""
        original_instruction = "Basic prompt without history"
        prompt = Prompt(original_instruction)

        simulator = UserSimulator(
            prompt=prompt, llm=mock_llm, agent_function=simple_agent
        )

        assert "conversation_history" in simulator.prompt.instruction
        assert simulator.prompt.instruction != original_instruction

    def test_user_simulator_prompt_already_has_history(self, mock_llm, simple_agent):
        """Test prompt with conversation_history placeholder is not modified."""
        original_instruction = "Respond based on {conversation_history}"
        prompt = Prompt(original_instruction)

        simulator = UserSimulator(
            prompt=prompt, llm=mock_llm, agent_function=simple_agent
        )

        # Should not modify prompt that already has conversation_history
        assert simulator.prompt.instruction == original_instruction

    def test_default_stopping_criteria_method(
        self, mock_prompt, mock_llm, simple_agent
    ):
        """Test default stopping criteria method."""
        simulator = UserSimulator(
            prompt=mock_prompt, llm=mock_llm, agent_function=simple_agent, max_turns=3
        )

        history = ConversationHistory()

        # Below max turns
        assert not simulator._default_stopping_criteria(history)

        # At max turns
        for i in range(3):
            history.add_message("user", f"msg {i}")
        assert simulator._default_stopping_criteria(history)

    def test_should_stop_conversation_exception_handling(
        self, mock_prompt, mock_llm, simple_agent
    ):
        """Test that stopping criteria exceptions are handled gracefully."""

        def failing_stop_criteria(history):
            raise Exception("Criteria failed")

        simulator = UserSimulator(
            prompt=mock_prompt,
            llm=mock_llm,
            agent_function=simple_agent,
            stopping_criteria=failing_stop_criteria,
        )

        history = ConversationHistory()
        # Should return True (stop conversation) when criteria fails
        assert simulator._should_stop_conversation(history) is True

    def test_format_conversation_for_prompt(self, mock_prompt, mock_llm, simple_agent):
        """Test conversation formatting for prompts."""
        simulator = UserSimulator(
            prompt=mock_prompt, llm=mock_llm, agent_function=simple_agent
        )

        # Empty history
        history = ConversationHistory()
        formatted = simulator._format_conversation_for_prompt(history)
        assert formatted == "No previous conversation."

        # With messages
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there")
        formatted = simulator._format_conversation_for_prompt(history)
        assert "User: Hello" in formatted
        assert "Assistant: Hi there" in formatted

    def test_format_conversation_complex_content(
        self, mock_prompt, mock_llm, simple_agent
    ):
        """Test conversation formatting with complex content types."""
        simulator = UserSimulator(
            prompt=mock_prompt, llm=mock_llm, agent_function=simple_agent
        )

        history = ConversationHistory()
        history.add_message("user", {"type": "text", "content": "complex message"})
        history.add_message("assistant", ["item1", "item2"])

        formatted = simulator._format_conversation_for_prompt(history)
        assert "User:" in formatted
        assert "Assistant:" in formatted

    def test_run_conversation_basic(self, mock_prompt, mock_llm, simple_agent):
        """Test running a basic conversation."""
        # Mock LLM responses
        mock_llm.generate.side_effect = [
            UserSimulatorResponse(content="User message 1", should_continue=True),
            UserSimulatorResponse(content="User message 2", should_continue=False),
        ]

        simulator = UserSimulator(
            prompt=mock_prompt, llm=mock_llm, agent_function=simple_agent, max_turns=10
        )

        history = simulator.run()

        # Should have messages from both user and assistant
        assert len(history.messages) > 0
        # Verify LLM was called
        assert mock_llm.generate.called

    def test_run_conversation_with_initial_message(
        self, mock_prompt, mock_llm, simple_agent
    ):
        """Test running conversation with initial message."""
        mock_llm.generate.return_value = UserSimulatorResponse(
            content="Response", should_continue=False
        )

        simulator = UserSimulator(
            prompt=mock_prompt, llm=mock_llm, agent_function=simple_agent
        )

        initial_msg = {"role": "user", "content": "Initial question"}
        history = simulator.run(initial_message=initial_msg)

        # Should start with initial message
        assert len(history.messages) >= 1
        assert history.messages[0].content == "Initial question"

    def test_run_conversation_agent_error_handling(self, mock_prompt, mock_llm):
        """Test handling of agent function errors."""

        def failing_agent(query, history):
            raise Exception("Agent failed")

        mock_llm.generate.return_value = UserSimulatorResponse(
            content="User msg", should_continue=False
        )

        simulator = UserSimulator(
            prompt=mock_prompt, llm=mock_llm, agent_function=failing_agent
        )

        initial_msg = {"role": "user", "content": "Test"}
        history = simulator.run(initial_message=initial_msg)

        # Should handle error gracefully
        assert len(history.messages) >= 2  # Initial + error response
        error_msg = next(
            (msg for msg in history.messages if "Error:" in str(msg.content)), None
        )
        assert error_msg is not None

    def test_run_conversation_different_agent_response_types(
        self, mock_prompt, mock_llm
    ):
        """Test handling different agent response formats."""
        responses = [
            "String response",
            {"content": "Dict response"},
            {"content": "Dict with role", "role": "assistant"},
        ]
        response_idx = 0

        def varied_agent(query, history):
            nonlocal response_idx
            response = responses[response_idx % len(responses)]
            response_idx += 1
            return response

        mock_llm.generate.return_value = UserSimulatorResponse(
            content="User", should_continue=False
        )

        simulator = UserSimulator(
            prompt=mock_prompt, llm=mock_llm, agent_function=varied_agent
        )

        # Test with multiple calls to exercise different response types
        for i in range(3):
            initial_msg = {"role": "user", "content": f"Test {i}"}
            history = simulator.run(initial_message=initial_msg)
            assert len(history.messages) >= 2  # At least user + agent response

    def test_custom_stopping_criteria(self, mock_prompt, mock_llm, simple_agent):
        """Test custom stopping criteria."""

        def custom_stop(history):
            return len(history.messages) >= 4

        mock_llm.generate.return_value = UserSimulatorResponse(
            content="Continue", should_continue=True
        )

        simulator = UserSimulator(
            prompt=mock_prompt,
            llm=mock_llm,
            agent_function=simple_agent,
            stopping_criteria=custom_stop,
            max_turns=10,  # Higher than custom criteria
        )

        history = simulator.run()

        # Should stop at 4 messages due to custom criteria
        assert len(history.messages) >= 4
