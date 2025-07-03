import pytest
from ragas.messages import ToolCall, AIMessage, HumanMessage
from ragas import MultiTurnSample
from ragas.metrics import ToolCallF1

metric = ToolCallF1()


def make_sample(expected, predicted):
    return MultiTurnSample(
        user_input=[
            HumanMessage(content="What is the weather in Paris?"),
            AIMessage(
                content="Let me check the weather forecast", tool_calls=predicted
            ),
        ],
        reference_tool_calls=expected,
        reference="Expected correct weather tool call",
    )


@pytest.mark.asyncio
async def test_tool_call_f1_full_match():
    expected = [ToolCall(name="WeatherForecast", args={"location": "Paris"})]
    predicted = [ToolCall(name="WeatherForecast", args={"location": "Paris"})]
    sample = make_sample(expected, predicted)
    score = await metric._multi_turn_ascore(sample)
    assert score == 1.0


@pytest.mark.asyncio
async def test_tool_call_f1_partial_match():
    expected = [
        ToolCall(name="WeatherForecast", args={"location": "Paris"}),
        ToolCall(name="UVIndex", args={"location": "Paris"}),
    ]
    predicted = [ToolCall(name="WeatherForecast", args={"location": "Paris"})]
    sample = make_sample(expected, predicted)
    score = await metric._multi_turn_ascore(sample)
    assert round(score, 2) == 0.67


@pytest.mark.asyncio
async def test_tool_call_f1_no_match():
    expected = [ToolCall(name="WeatherForecast", args={"location": "Paris"})]
    predicted = [ToolCall(name="AirQuality", args={"location": "Paris"})]
    sample = make_sample(expected, predicted)
    score = await metric._multi_turn_ascore(sample)
    assert score == 0.0


@pytest.mark.asyncio
async def test_tool_call_f1_extra_call():
    expected = [ToolCall(name="WeatherForecast", args={"location": "Paris"})]
    predicted = [
        ToolCall(name="WeatherForecast", args={"location": "Paris"}),
        ToolCall(name="AirQuality", args={"location": "Paris"}),
    ]
    sample = make_sample(expected, predicted)
    score = await metric._multi_turn_ascore(sample)
    assert round(score, 2) == 0.67
