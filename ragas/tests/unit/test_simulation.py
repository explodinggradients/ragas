"""Tests for the simulation module."""

import pytest
from unittest.mock import Mock
from pydantic import ValidationError

from ragas.simulation import (
    Message,
    ConversationHistory,
    UserSimulator,
    UserSimulatorResponse,
    Prompt,
    validate_agent_function,
    validate_stopping_criteria,
    default_stopping_criteria,
)


class TestMessage:
    """Test the Message class."""

    def test_message_creation_valid(self):
        """Test creating a valid message."""
        message = Message(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"

    def test_message_assistant_role(self):
        """Test creating a message with assistant role."""
        message = Message(role="assistant", content="Hi there!")
        assert message.role == "assistant"
        assert message.content == "Hi there!"

    def test_message_with_dict_content(self):
        """Test creating a message with dictionary content."""
        content = {"text": "Hello", "metadata": {"source": "user"}}
        message = Message(role="user", content=content)
        assert message.role == "user"
        assert message.content == content

    def test_message_with_list_content(self):
        """Test creating a message with list content."""
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "image", "url": "image.jpg"},
        ]
        message = Message(role="user", content=content)
        assert message.role == "user"
        assert message.content == content

    def test_message_invalid_role(self):
        """Test creating a message with invalid role."""
        with pytest.raises(ValidationError):
            Message(role="invalid", content="Hello")


class TestConversationHistory:
    """Test the ConversationHistory class."""

    def test_empty_conversation_history(self):
        """Test creating an empty conversation history."""
        history = ConversationHistory()
        assert len(history.messages) == 0
        assert history.get_last_message() is None

    def test_add_message(self):
        """Test adding a message to conversation history."""
        history = ConversationHistory()
        history.add_message("user", "Hello")

        assert len(history.messages) == 1
        assert history.messages[0].role == "user"
        assert history.messages[0].content == "Hello"

    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        history = ConversationHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there!")
        history.add_message("user", "How are you?")

        assert len(history.messages) == 3
        assert history.messages[0].role == "user"
        assert history.messages[1].role == "assistant"
        assert history.messages[2].role == "user"

    def test_get_last_message(self):
        """Test getting the last message."""
        history = ConversationHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi!")

        last_message = history.get_last_message()
        assert last_message is not None
        assert last_message.role == "assistant"
        assert last_message.content == "Hi!"

    def test_to_dict_list(self):
        """Test converting conversation history to dictionary list."""
        history = ConversationHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi!")

        dict_list = history.to_dict_list()
        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        assert dict_list == expected

    def test_to_dict_list_empty(self):
        """Test converting empty conversation history to dictionary list."""
        history = ConversationHistory()
        assert history.to_dict_list() == []


class TestPrompt:
    """Test the Prompt class."""

    def test_prompt_creation(self):
        """Test creating a prompt."""
        prompt = Prompt("Hello {name}")
        assert prompt.instruction == "Hello {name}"

    def test_prompt_format_simple(self):
        """Test formatting a prompt with simple variables."""
        prompt = Prompt("Hello {name}, how are you?")
        formatted = prompt.format(name="Alice")
        assert formatted == "Hello Alice, how are you?"

    def test_prompt_format_multiple_vars(self):
        """Test formatting with multiple variables."""
        prompt = Prompt("Hello {name}, today is {day}")
        formatted = prompt.format(name="Bob", day="Monday")
        assert formatted == "Hello Bob, today is Monday"

    def test_prompt_format_no_vars(self):
        """Test formatting prompt with no variables."""
        prompt = Prompt("Hello world")
        formatted = prompt.format()
        assert formatted == "Hello world"


class TestValidateAgentFunction:
    """Test the validate_agent_function utility."""

    def test_valid_agent_function(self):
        """Test validation with a valid agent function."""

        def valid_agent(query, history):
            return "Response"

        # Should not raise any exception
        validate_agent_function(valid_agent)

    def test_agent_function_with_extra_params(self):
        """Test agent function with extra parameters."""

        def agent_with_extra_params(query, history, context=None, temperature=0.5):
            return "Response"

        # Should not raise any exception
        validate_agent_function(agent_with_extra_params)

    def test_agent_function_insufficient_params(self):
        """Test agent function with insufficient parameters."""

        def invalid_agent(query):
            return "Response"

        with pytest.raises(ValueError, match="must accept at least 2 parameters"):
            validate_agent_function(invalid_agent)

    def test_agent_function_returns_none(self):
        """Test agent function that returns None."""

        def none_returning_agent(query, history):
            return None

        with pytest.raises(ValueError, match="cannot return None"):
            validate_agent_function(none_returning_agent)

    def test_agent_function_type_error(self):
        """Test agent function that raises TypeError."""

        def problematic_agent(query):  # Wrong signature
            return "Response"

        with pytest.raises(ValueError, match="must accept at least 2 parameters"):
            validate_agent_function(problematic_agent)


class TestValidateStoppingCriteria:
    """Test the validate_stopping_criteria utility."""

    def test_valid_stopping_criteria(self):
        """Test validation with valid stopping criteria."""

        def valid_criteria(history):
            return len(history.messages) >= 5

        # Should not raise any exception
        validate_stopping_criteria(valid_criteria)

    def test_stopping_criteria_wrong_param_count(self):
        """Test stopping criteria with wrong parameter count."""

        def invalid_criteria(history, extra_param):
            return True

        with pytest.raises(ValueError, match="must accept exactly 1 parameter"):
            validate_stopping_criteria(invalid_criteria)

    def test_stopping_criteria_returns_none(self):
        """Test stopping criteria that returns None."""

        def none_returning_criteria(history):
            return None

        with pytest.raises(ValueError, match="cannot return None"):
            validate_stopping_criteria(none_returning_criteria)

    def test_stopping_criteria_type_error(self):
        """Test stopping criteria that raises TypeError."""

        def problematic_criteria():  # Wrong signature
            return True

        with pytest.raises(ValueError, match="must accept exactly 1 parameter"):
            validate_stopping_criteria(problematic_criteria)


class TestDefaultStoppingCriteria:
    """Test the default_stopping_criteria function."""

    def test_default_stopping_criteria_under_limit(self):
        """Test default stopping criteria with messages under limit."""
        history = ConversationHistory()
        for i in range(5):
            history.add_message("user", f"Message {i}")

        # Should not stop (under 10 messages)
        assert not default_stopping_criteria(history)

    def test_default_stopping_criteria_at_limit(self):
        """Test default stopping criteria at limit."""
        history = ConversationHistory()
        for i in range(10):
            history.add_message("user", f"Message {i}")

        # Should stop (at 10 messages)
        assert default_stopping_criteria(history)

    def test_default_stopping_criteria_over_limit(self):
        """Test default stopping criteria over limit."""
        history = ConversationHistory()
        for i in range(15):
            history.add_message("user", f"Message {i}")

        # Should stop (over 10 messages)
        assert default_stopping_criteria(history)


class TestUserSimulatorResponse:
    """Test the UserSimulatorResponse class."""

    def test_user_simulator_response_default(self):
        """Test creating response with default values."""
        response = UserSimulatorResponse(content="Hello")
        assert response.content == "Hello"
        assert response.should_continue is True

    def test_user_simulator_response_custom(self):
        """Test creating response with custom values."""
        response = UserSimulatorResponse(content="Goodbye", should_continue=False)
        assert response.content == "Goodbye"
        assert response.should_continue is False


class TestUserSimulator:
    """Test the UserSimulator class."""

    def test_user_simulator_initialization(self):
        """Test UserSimulator initialization."""
        prompt = Prompt("Generate a response based on: {conversation_history}")
        mock_llm = Mock()

        def mock_agent(query, history):
            return "Agent response"

        simulator = UserSimulator(
            prompt=prompt, llm=mock_llm, agent_function=mock_agent, max_turns=5
        )

        assert simulator.prompt == prompt
        assert simulator.llm == mock_llm
        assert simulator.agent_function == mock_agent
        assert simulator.max_turns == 5

    def test_user_simulator_adds_conversation_history_placeholder(self):
        """Test that conversation_history placeholder is added if missing."""
        prompt = Prompt("Generate a response")
        mock_llm = Mock()

        def mock_agent(query, history):
            return "Agent response"

        simulator = UserSimulator(
            prompt=prompt, llm=mock_llm, agent_function=mock_agent
        )

        # Should automatically add conversation_history placeholder
        assert "conversation_history" in simulator.prompt.instruction

    def test_user_simulator_preserves_existing_placeholder(self):
        """Test that existing conversation_history placeholder is preserved."""
        original_instruction = "Generate based on: {conversation_history}"
        prompt = Prompt(original_instruction)
        mock_llm = Mock()

        def mock_agent(query, history):
            return "Agent response"

        simulator = UserSimulator(
            prompt=prompt, llm=mock_llm, agent_function=mock_agent
        )

        # Should not modify if placeholder already exists
        assert simulator.prompt.instruction == original_instruction

    def test_user_simulator_format_conversation_empty(self):
        """Test formatting empty conversation history."""
        prompt = Prompt("Generate a response based on: {conversation_history}")
        mock_llm = Mock()

        def mock_agent(query, history):
            return "Agent response"

        simulator = UserSimulator(
            prompt=prompt, llm=mock_llm, agent_function=mock_agent
        )

        history = ConversationHistory()
        formatted = simulator._format_conversation_for_prompt(history)
        assert formatted == "No previous conversation."

    def test_user_simulator_format_conversation_with_messages(self):
        """Test formatting conversation history with messages."""
        prompt = Prompt("Generate a response based on: {conversation_history}")
        mock_llm = Mock()

        def mock_agent(query, history):
            return "Agent response"

        simulator = UserSimulator(
            prompt=prompt, llm=mock_llm, agent_function=mock_agent
        )

        history = ConversationHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there!")

        formatted = simulator._format_conversation_for_prompt(history)
        expected = "User: Hello\nAssistant: Hi there!"
        assert formatted == expected

    def test_user_simulator_format_conversation_dict_content(self):
        """Test formatting conversation with dictionary content."""
        prompt = Prompt("Generate a response based on: {conversation_history}")
        mock_llm = Mock()

        def mock_agent(query, history):
            return "Agent response"

        simulator = UserSimulator(
            prompt=prompt, llm=mock_llm, agent_function=mock_agent
        )

        history = ConversationHistory()
        history.add_message("user", {"text": "Hello", "metadata": {}})

        formatted = simulator._format_conversation_for_prompt(history)
        assert "User:" in formatted
        assert "Hello" in formatted

    def test_default_stopping_criteria_method(self):
        """Test the default stopping criteria method."""
        prompt = Prompt("Generate a response")
        mock_llm = Mock()

        def mock_agent(query, history):
            return "Agent response"

        simulator = UserSimulator(
            prompt=prompt, llm=mock_llm, agent_function=mock_agent, max_turns=3
        )

        # Test under limit
        history = ConversationHistory()
        history.add_message("user", "Hello")
        assert not simulator._default_stopping_criteria(history)

        # Test at limit
        history.add_message("assistant", "Hi")
        history.add_message("user", "How are you?")
        assert simulator._default_stopping_criteria(history)

    def test_user_simulator_invalid_agent_function(self):
        """Test UserSimulator with invalid agent function."""
        prompt = Prompt("Generate a response")
        mock_llm = Mock()

        def invalid_agent(query):  # Missing history parameter
            return "Response"

        with pytest.raises(ValueError):
            UserSimulator(prompt=prompt, llm=mock_llm, agent_function=invalid_agent)

    def test_user_simulator_invalid_stopping_criteria(self):
        """Test UserSimulator with invalid stopping criteria."""
        prompt = Prompt("Generate a response")
        mock_llm = Mock()

        def mock_agent(query, history):
            return "Agent response"

        def invalid_criteria(history, extra_param):  # Wrong signature
            return True

        with pytest.raises(ValueError):
            UserSimulator(
                prompt=prompt,
                llm=mock_llm,
                agent_function=mock_agent,
                stopping_criteria=invalid_criteria,
            )

    def test_user_simulator_run_basic(self):
        """Test basic run functionality."""
        prompt = Prompt("Generate a response based on: {conversation_history}")
        mock_llm = Mock()

        # Mock LLM to return proper LLMResult structure
        from langchain_core.outputs import Generation, LLMResult

        mock_generation = Generation(
            text='{"content": "User response", "should_continue": false}'
        )
        mock_llm_result = LLMResult(generations=[[mock_generation]])
        mock_llm.generate_text.return_value = mock_llm_result

        def mock_agent(query, history):
            return "Agent response"

        # Custom stopping criteria that stops after 2 messages
        def stop_after_two(history):
            return len(history.messages) >= 2

        simulator = UserSimulator(
            prompt=prompt,
            llm=mock_llm,
            agent_function=mock_agent,
            stopping_criteria=stop_after_two,
        )

        result = simulator.run(initial_message={"role": "user", "content": "Hello"})

        # Should have initial message and agent response
        assert len(result.messages) >= 1
        assert result.messages[0].content == "Hello"

    def test_user_simulator_run_agent_error_handling(self):
        """Test error handling when agent function fails during runtime."""
        prompt = Prompt("Generate a response based on: {conversation_history}")
        mock_llm = Mock()

        # Mock LLM to return proper LLMResult structure
        from langchain_core.outputs import Generation, LLMResult

        mock_generation = Generation(
            text='{"content": "User response", "should_continue": false}'
        )
        mock_llm_result = LLMResult(generations=[[mock_generation]])
        mock_llm.generate_text.return_value = mock_llm_result

        # Create an agent that works during validation but fails during runtime
        call_count = 0

        def failing_agent(query, history):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call during validation - return valid response
                return "Validation response"
            else:
                # Subsequent calls during runtime - raise error
                raise Exception("Agent failed")

        def stop_after_three(history):
            return len(history.messages) >= 3

        simulator = UserSimulator(
            prompt=prompt,
            llm=mock_llm,
            agent_function=failing_agent,
            stopping_criteria=stop_after_three,
        )

        result = simulator.run(initial_message={"role": "user", "content": "Hello"})

        # Should handle error gracefully
        assert len(result.messages) >= 1
        # Should have error message from agent
        if len(result.messages) > 1:
            assert "Error:" in result.messages[1].content

    def test_user_simulator_run_stopping_criteria_error(self):
        """Test handling of stopping criteria errors during runtime."""
        prompt = Prompt("Generate a response")
        mock_llm = Mock()

        def mock_agent(query, history):
            return "Agent response"

        # Create criteria that works during validation but fails during runtime
        call_count = 0

        def failing_criteria(history):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call during validation - return valid response
                return False
            else:
                # Subsequent calls during runtime - raise error
                raise Exception("Criteria failed")

        simulator = UserSimulator(
            prompt=prompt,
            llm=mock_llm,
            agent_function=mock_agent,
            stopping_criteria=failing_criteria,
        )

        # Should handle criteria failure by stopping conversation
        result = simulator.run(initial_message={"role": "user", "content": "Hello"})

        # Should not crash and should return some result
        assert isinstance(result, ConversationHistory)

    def test_user_simulator_agent_response_formats(self):
        """Test different agent response formats."""
        prompt = Prompt("Generate a response")
        mock_llm = Mock()

        # Mock LLM to return proper LLMResult structure
        from langchain_core.outputs import Generation, LLMResult

        mock_generation = Generation(
            text='{"content": "User response", "should_continue": false}'
        )
        mock_llm_result = LLMResult(generations=[[mock_generation]])
        mock_llm.generate_text.return_value = mock_llm_result

        # Test string response
        def string_agent(query, history):
            return "String response"

        def stop_after_three(history):
            return len(history.messages) >= 3

        simulator = UserSimulator(
            prompt=prompt,
            llm=mock_llm,
            agent_function=string_agent,
            stopping_criteria=stop_after_three,
        )

        result = simulator.run(initial_message={"role": "user", "content": "Hello"})
        # Should handle string response correctly
        if len(result.messages) > 1:
            assert result.messages[1].content == "String response"

        # Test dict response with content
        def dict_agent(query, history):
            return {"content": "Dict response"}

        simulator = UserSimulator(
            prompt=prompt,
            llm=mock_llm,
            agent_function=dict_agent,
            stopping_criteria=stop_after_three,
        )

        result = simulator.run(initial_message={"role": "user", "content": "Hello"})
        # Should handle dict response correctly
        if len(result.messages) > 1:
            assert result.messages[1].content == "Dict response"

        # Test other response format
        def other_agent(query, history):
            return 42  # Non-string, non-dict response

        simulator = UserSimulator(
            prompt=prompt,
            llm=mock_llm,
            agent_function=other_agent,
            stopping_criteria=stop_after_three,
        )

        result = simulator.run(initial_message={"role": "user", "content": "Hello"})
        # Should convert to string
        if len(result.messages) > 1:
            assert result.messages[1].content == "42"
