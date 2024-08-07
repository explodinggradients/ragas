from __future__ import annotations

import typing as t

import pytest


def test_debug_tracking_flag():
    import os

    from ragas._analytics import RAGAS_DEBUG_TRACKING

    assert os.environ.get(RAGAS_DEBUG_TRACKING, "").lower() == "true"


def test_base_event():
    from ragas._analytics import BaseEvent

    be = BaseEvent(event_type="evaluation")
    assert isinstance(dict(be).get("event_type"), str)
    assert isinstance(dict(be).get("user_id"), str)


def test_evaluation_event():
    from ragas._analytics import EvaluationEvent

    evaluation_event = EvaluationEvent(
        event_type="evaluation",
        metrics=["harmfulness"],
        num_rows=1,
        evaluation_mode="",
        language="english",
        in_ci=True,
    )

    payload = dict(evaluation_event)
    assert isinstance(payload.get("user_id"), str)
    assert isinstance(payload.get("evaluation_mode"), str)
    assert isinstance(payload.get("metrics"), list)
    assert isinstance(payload.get("language"), str)
    assert isinstance(payload.get("in_ci"), bool)


def setup_user_id_filepath(tmp_path, monkeypatch):
    # setup
    def user_data_dir_patch(appname, roaming=True) -> str:
        return str(tmp_path / appname)

    import ragas._analytics
    from ragas._analytics import USER_DATA_DIR_NAME

    monkeypatch.setattr(ragas._analytics, "user_data_dir", user_data_dir_patch)
    userid_filepath = tmp_path / USER_DATA_DIR_NAME / "uuid.json"

    return userid_filepath


def test_write_to_file(tmp_path, monkeypatch):
    userid_filepath = setup_user_id_filepath(tmp_path, monkeypatch)

    # check if file created if not existing
    assert not userid_filepath.exists()
    import json

    from ragas._analytics import get_userid

    # clear LRU cache since its created in setup for the above test
    get_userid.cache_clear()

    userid = get_userid()
    assert userid_filepath.exists()
    with open(userid_filepath, "r") as f:
        assert userid == json.load(f)["userid"]

    assert not (tmp_path / "uuid.json").exists()

    # del file and check if LRU cache is working
    userid_filepath.unlink()
    assert not userid_filepath.exists()
    userid_cached = get_userid()
    assert userid == userid_cached


def test_load_userid_from_json_file(tmp_path, monkeypatch):
    userid_filepath = setup_user_id_filepath(tmp_path, monkeypatch)
    assert not userid_filepath.exists()

    # create uuid.json file
    userid_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(userid_filepath, "w") as f:
        import json

        json.dump({"userid": "test-userid"}, f)

    from ragas._analytics import get_userid

    # clear LRU cache since its created in setup for the above test
    get_userid.cache_clear()

    assert get_userid() == "test-userid"


def test_testset_generation_tracking(monkeypatch):
    import ragas._analytics as analyticsmodule
    from ragas._analytics import TestsetGenerationEvent, track
    from ragas.testset.evolutions import multi_context, reasoning, simple

    distributions = {simple: 0.5, multi_context: 0.3, reasoning: 0.2}

    testset_event_payload = TestsetGenerationEvent(
        event_type="testset_generation",
        evolution_names=[e.__class__.__name__.lower() for e in distributions],
        evolution_percentages=[distributions[e] for e in distributions],
        num_rows=10,
        language="english",
    )

    assert dict(testset_event_payload)["evolution_names"] == [
        "simpleevolution",
        "multicontextevolution",
        "reasoningevolution",
    ]

    assert dict(testset_event_payload)["evolution_percentages"] == [0.5, 0.3, 0.2]

    # just in the case you actually want to check if tracking is working in the
    # dashboard
    if False:
        monkeypatch.setattr(analyticsmodule, "do_not_track", lambda: False)
        monkeypatch.setattr(analyticsmodule, "_usage_event_debugging", lambda: False)
        track(testset_event_payload)


def test_was_completed(monkeypatch):
    from ragas._analytics import IsCompleteEvent, track_was_completed

    event_properties_list: t.List[IsCompleteEvent] = []

    def echo_track(event_properties):
        event_properties_list.append(event_properties)

    monkeypatch.setattr("ragas._analytics.track", echo_track)

    @track_was_completed
    def test(raise_error=True):
        if raise_error:
            raise ValueError("test")
        else:
            pass

    with pytest.raises(ValueError):
        test(raise_error=True)

    assert event_properties_list[-1].event_type == "test"
    assert event_properties_list[-1].is_completed is False

    test(raise_error=False)

    assert event_properties_list[-1].event_type == "test"
    assert event_properties_list[-1].is_completed is True
