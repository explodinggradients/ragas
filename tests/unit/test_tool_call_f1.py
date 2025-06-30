import pytest
from ragas.messages import ToolCall, AIMessage, HumanMessage
from ragas.types import MultiTurnSample
from ragas.metrics._tool_call_f1 import ToolCallF1

metric = ToolCallF1()

def make_sample(expected, predicted):
    return MultiTurnSample(
        user_input=[HumanMessage(content="What is the weather in Paris?")],
        agent_messages=[
            AIMessage(
                content="Let me check the weather forecast",
                tool_calls=predicted
            )
        ],
        reference_tool_calls=expected,
        reference="Expected correct weather tool call"
    )

def test_tool_call_f1_full_match():
    expected = [
        ToolCall(name="WeatherForecast", parameters={"location": "Paris"})
    ]
    predicted = [
        ToolCall(name="WeatherForecast", parameters={"location": "Paris"})
    ]
    sample = make_sample(expected, predicted)
    score = pytest.run(metric._multi_turn_ascore(sample))
    assert score == 1.0

def test_tool_call_f1_partial_match():
    expected = [
        ToolCall(name="WeatherForecast", parameters={"location": "Paris"}),
        ToolCall(name="UVIndex", parameters={"location": "Paris"})
    ]
    predicted = [
        ToolCall(name="WeatherForecast", parameters={"location": "Paris"})
    ]
    sample = make_sample(expected, predicted)
    score = pytest.run(metric._multi_turn_ascore(sample))
    assert round(score, 2) == 0.67

def test_tool_call_f1_no_match():
    expected = [
        ToolCall(name="WeatherForecast", parameters={"location": "Paris"})
    ]
    predicted = [
        ToolCall(name="AirQuality", parameters={"location": "Paris"})
    ]
    sample = make_sample(expected, predicted)
    score = pytest.run(metric._multi_turn_ascore(sample))
    assert score == 0.0

def test_tool_call_f1_extra_call():
    expected = [
        ToolCall(name="WeatherForecast", parameters={"location": "Paris"})
    ]
    predicted = [
        ToolCall(name="WeatherForecast", parameters={"location": "Paris"}),
        ToolCall(name="AirQuality", parameters={"location": "Paris"})
    ]
    sample = make_sample(expected, predicted)
    score = pytest.run(metric._multi_turn_ascore(sample))
    assert round(score, 2) == 0.67
